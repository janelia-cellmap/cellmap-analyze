# %%
import logging
from pathlib import Path

import numpy as np
import worldarray as wa
from funlib.geometry import Coordinate, Roi

from cellmap_analyze.util.io_util import is_remote_path
from cellmap_analyze.util.voxel_size_utils import scale_voxel_size_to_integers

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _concurrency_context(concurrency_limit):
    """The per-handle context cellmap-analyze has always used: data copies and file IO
    capped, network request pools left at tensorstore's defaults."""
    if not concurrency_limit:
        return None
    return wa.capped_context(concurrency=concurrency_limit, network=False)


def _open_world_array(path, target_voxel_size, context, mode):
    """The array at ``path``; for a multiscale group, its first (finest) level, or the
    coarsest level no coarser than ``target_voxel_size`` when one is given (the first
    when every level is coarser), at that level's own voxel size."""
    if target_voxel_size is not None:
        return wa.open(
            path,
            voxel_size=tuple(float(v) for v in target_voxel_size),
            level_mode="floor",
            on_mismatch="keep",
            context=context,
            mode=mode,
        )
    node = wa.detect(path)
    if node.kind == "array" or (node.format == "precomputed" and node.scale_key is not None):
        return wa.open(path, context=context, mode=mode)
    return wa.open(path, level="first", context=context, mode=mode)


class ImageDataInterface:
    """cellmap-analyze's view of a dataset: worldarray for every read and write and for
    where the data sits, plus the integer-scaled frame (funlib ``Coordinate``/``Roi``)
    the blockwise code works in.

    In the frame, every length is nm times ``voxel_size_scale_factor``, an integer chosen
    so the voxel size and the corner of voxel 0 are both whole.
    """

    def __init__(
        self,
        dataset_path,
        mode="r",
        output_voxel_size=None,
        custom_fill_value=None,
        concurrency_limit=1,
        chunk_shape=None,
        max_retries=10,
        timeout=5,
        interpolation_order=0,
        target_voxel_size=None,
    ):
        dataset_path = str(dataset_path)
        if not is_remote_path(dataset_path) and not dataset_path.startswith("precomputed://"):
            dataset_path = str(Path(dataset_path).resolve())
        self.mode = mode
        self.concurrency_limit = concurrency_limit
        self._wa = _open_world_array(
            dataset_path,
            target_voxel_size,
            _concurrency_context(concurrency_limit),
            mode,
        )
        self._wa_concurrency = concurrency_limit
        # the opened level's own path (wa.path is the path as given, e.g. the group)
        self.path = self._wa.meta.path.rstrip("/")
        self.dtype = self._wa.dtype
        self.shape = tuple(self._wa.shape)
        self.chunk_shape = Coordinate(self._wa.meta.chunk_shape)
        if chunk_shape is not None:
            self.chunk_shape = Coordinate(chunk_shape)

        # True nm, spatial axes, zyx; the corner of voxel 0 (translation - voxel_size / 2).
        self.original_voxel_size = tuple(self._wa.voxel_size)
        self._raw_offset = tuple(self._wa.corner)
        self._spatial_shape = tuple(self._wa.shape[i] for i in self._wa.spatial)
        _, scale_factor = scale_voxel_size_to_integers(
            self.original_voxel_size + self._raw_offset
        )
        self._set_frame(scale_factor)

        self.custom_fill_value = custom_fill_value
        if output_voxel_size is not None:
            self.output_voxel_size = output_voxel_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.interpolation_order = interpolation_order

    def _set_frame(self, scale_factor):
        self.voxel_size_scale_factor = scale_factor
        self.voxel_size = Coordinate(int(round(v * scale_factor)) for v in self.original_voxel_size)
        self.roi = Roi(
            Coordinate(int(round(c * scale_factor)) for c in self._raw_offset),
            Coordinate(self._spatial_shape) * self.voxel_size,
        )
        self.offset = self.roi.offset
        self.output_voxel_size = self.voxel_size

    def rescale_to_factor(self, new_scale_factor):
        """Re-express the frame with another integer factor, so several datasets can
        share one frame (contact sites, masks, measure)."""
        if new_scale_factor != self.voxel_size_scale_factor:
            self._set_frame(new_scale_factor)

    def _world_array(self):
        if self._wa_concurrency != self.concurrency_limit:
            # e.g. dask_util.rescale_idi_concurrency gave this worker its fair share
            self._wa = self._wa.reopen(context=_concurrency_context(self.concurrency_limit))
            self._wa_concurrency = self.concurrency_limit
        return self._wa

    def to_ndarray_ts(self, roi=None, return_valid_mask=False):
        """The data in ``roi`` (frame units; the whole dataset when None) at
        ``output_voxel_size``, padded with ``custom_fill_value`` (0 by default) where it
        leaves the data. With ``return_valid_mask``, also True where the data was."""
        factor = self.voxel_size_scale_factor
        if roi is None:
            roi = self.roi
        box = (
            tuple(b / factor for b in roi.begin),
            tuple(e / factor for e in roi.end),
        )
        voxel_size = None
        if Coordinate(self.output_voxel_size) != self.voxel_size:
            voxel_size = tuple(v / factor for v in self.output_voxel_size)
        return self._world_array().read(
            box,
            voxel_size=voxel_size,
            padding=0 if self.custom_fill_value is None else self.custom_fill_value,
            method="nearest" if self.interpolation_order == 0 else "linear",
            mask=return_valid_mask,
            retries=wa.Retry(max_attempts=self.max_retries, timeout=self.timeout),
            # callers' boxes sit on the grid they resample onto (the other dataset's)
            anchor=box[0] if voxel_size is not None else None,
        )

    def write(self, roi, data):
        """Write ``data`` (an array, or one value for the whole box) into ``roi`` (frame
        units), which must cover whole chunks (or end at the array's edge)."""
        begin = (roi.begin - self.roi.begin) / self.voxel_size
        shape = roi.shape / self.voxel_size
        data = np.broadcast_to(np.asarray(data, dtype=self.dtype), tuple(shape))
        self._world_array().write_index(tuple(begin), tuple(begin + shape), data)

    def chunk_key(self, index):
        """The storage key, relative to the array, of the chunk at ``index``; a shorter
        index gives the prefix shared by the chunks under it."""
        return self._wa.chunk_key(index)

# %%
import logging
from pathlib import Path

import numpy as np
import worldarray as wa

from cellmap_analyze.util.grid_util import make_grid
from cellmap_analyze.util.io_util import is_remote_path

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
    """cellmap-analyze's view of a dataset, through worldarray.

    Boxes are voxel indices, ``[begin, end)`` on the spatial axes (zyx). Reads are on
    ``output_grid``: the dataset's own ``grid`` unless it is set to another (the
    operation's grid), onto which the data is then resampled. Writes are on ``grid``.
    ``voxel_size`` and ``corner`` (of voxel 0) are in nm.
    """

    def __init__(
        self,
        dataset_path,
        mode="r",
        custom_fill_value=None,
        concurrency_limit=1,
        chunk_shape=None,
        max_retries=10,
        timeout=5,
        interpolation_order=0,
        target_voxel_size=None,
        output_grid=None,
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
        spatial = self._wa.spatial
        if chunk_shape is None:
            chunk_shape = [self._wa.meta.chunk_shape[i] for i in spatial]
        self.chunk_shape = tuple(int(c) for c in chunk_shape)
        self.voxel_size = tuple(self._wa.voxel_size)
        self.corner = tuple(self._wa.corner)
        self.grid = make_grid(
            [self.shape[i] for i in spatial], self.voxel_size, self.corner, self.chunk_shape
        )
        self.output_grid = self.grid if output_grid is None else output_grid

        self.custom_fill_value = custom_fill_value
        self.max_retries = max_retries
        self.timeout = timeout
        self.interpolation_order = interpolation_order

    def _world_array(self):
        if self._wa_concurrency != self.concurrency_limit:
            # e.g. dask_util.rescale_idi_concurrency gave this worker its fair share
            self._wa = self._wa.reopen(context=_concurrency_context(self.concurrency_limit))
            self._wa_concurrency = self.concurrency_limit
        return self._wa

    def to_ndarray_ts(self, begin=None, end=None, return_valid_mask=False):
        """The voxels ``[begin, end)`` of ``output_grid`` (all of them when None),
        padded with ``custom_fill_value`` (0 by default) where they leave the data. With
        ``return_valid_mask``, also True where the data was."""
        grid = self.output_grid
        if begin is None:
            begin = (0,) * len(grid.shape)
        if end is None:
            end = grid.shape
        resample = not np.allclose(grid.voxel_size, self.voxel_size, rtol=0, atol=1e-9)
        return self._world_array().read(
            wa.Box.from_index(begin, end, grid.voxel_size, grid.translation),
            voxel_size=grid.voxel_size if resample else None,
            padding=0 if self.custom_fill_value is None else self.custom_fill_value,
            method="nearest" if self.interpolation_order == 0 else "linear",
            mask=return_valid_mask,
            retries=wa.Retry(max_attempts=self.max_retries, timeout=self.timeout),
            # read onto the output grid's voxels, wherever this dataset's sit
            anchor=grid.corner if resample else None,
        )

    def write(self, begin, end, data):
        """Write ``data`` (an array, or one value for the whole box) into voxels
        ``[begin, end)`` of ``grid``, which must cover whole chunks (or end at the
        array's edge)."""
        shape = tuple(int(e - b) for b, e in zip(begin, end))
        data = np.broadcast_to(np.asarray(data, dtype=self.dtype), shape)
        self._world_array().write_index(tuple(begin), tuple(end), data)

    def chunk_key(self, index):
        """The storage key, relative to the array, of the chunk at ``index``; a shorter
        index gives the prefix shared by the chunks under it."""
        return self._wa.chunk_key(index)

# %%
import logging
import os
from pathlib import Path

import tensorstore as ts
import worldarray as wa
from funlib.geometry import Coordinate, Roi

from cellmap_analyze.util.io_util import split_dataset_path
from cellmap_analyze.util.voxel_size_utils import scale_voxel_size_to_integers
from cellmap_analyze.util.zarr_io import open_dataset

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# worldarray's format names, as cellmap-analyze's filetype names
_FILETYPES = {"zarr2": "zarr", "zarr3": "zarr3", "n5": "n5", "precomputed": "neuroglancer_precomputed"}


def _detect_zarr_driver(dataset_path: str) -> str:
    """Detect the tensorstore driver for ``dataset_path``.

    Returns one of ``"zarr"``, ``"zarr3"``, ``"n5"``, or
    ``"neuroglancer_precomputed"``. Works for both local paths and
    remote URIs (``s3://`` / ``gs://`` / ``http(s)://``).

    Precomputed is detected by either the explicit ``precomputed://``
    URL prefix OR by probing for an ``info`` marker file (at the path
    or its parent, since precomputed paths often end in a scale key
    like ``/s0`` that's not a real subdirectory).
    """
    from cellmap_analyze.util.io_util import (
        is_precomputed_path,
        is_remote_path,
        path_dirname,
        path_join,
        read_json_path,
        strip_precomputed_prefix,
    )

    if is_precomputed_path(dataset_path):
        return "neuroglancer_precomputed"

    canonical = strip_precomputed_prefix(dataset_path)

    # Precomputed: ``info`` sits at the dataset root. Try the path itself
    # first, then its parent (covers paths like ``.../seg/s0`` where the
    # trailing segment is a scale key, not a real subdirectory).
    if read_json_path(path_join(canonical, "info")) is not None:
        return "neuroglancer_precomputed"
    parent = path_dirname(canonical)
    if parent and parent != canonical:
        if read_json_path(path_join(parent, "info")) is not None:
            return "neuroglancer_precomputed"

    if canonical.rfind(".n5") > canonical.rfind(".zarr"):
        return "n5"

    # Check for zarr v3 format marker (zarr.json at dataset level).
    # ``read_json_path`` returns None when missing -- exactly the
    # fall-through-to-v2 signal we want.
    zarr_json_path = path_join(canonical, "zarr.json")
    if is_remote_path(canonical):
        if read_json_path(zarr_json_path) is not None:
            return "zarr3"
    else:
        if os.path.exists(zarr_json_path):
            return "zarr3"
    return "zarr"


def open_ds_tensorstore(dataset_path: str, mode="r", concurrency_limit=None):
    """Open a tensorstore handle for ``dataset_path``.

    ``dataset_path`` may be a local filesystem path or a remote URI
    (``s3://``, ``gs://``, ``http(s)://``, ``file://``). Tensorstore's
    native kvstore drivers handle the transport -- no fsspec/s3fs.
    """
    from cellmap_analyze.util.io_util import (
        is_remote_path,
        kvstore_for_path,
        strip_precomputed_prefix,
    )

    if is_remote_path(dataset_path) and mode != "r":
        raise ValueError(
            f"remote write to {dataset_path!r} is not supported; "
            f"cellmap-analyze only supports reading from object storage."
        )

    # open with zarr, zarr3, n5, or neuroglancer_precomputed depending on format
    filetype = _detect_zarr_driver(dataset_path)
    if filetype == "neuroglancer_precomputed":
        if mode != "r":
            raise ValueError(
                "neuroglancer precomputed datasets are read-only"
            )
        from cellmap_analyze.util.precomputed_io import (
            open_precomputed_tensorstore,
        )
        return open_precomputed_tensorstore(dataset_path)

    dataset_path = strip_precomputed_prefix(dataset_path)
    kvstore, key = kvstore_for_path(dataset_path)
    # tensorstore expects the chunk key inside the kvstore. Append a
    # trailing slash so it's treated as a directory of chunks.
    kvstore["path"] = (key.rstrip("/") + "/") if key else ""
    if concurrency_limit:
        spec = {
            "driver": filetype,
            "context": {
                "data_copy_concurrency": {"limit": concurrency_limit},
                "file_io_concurrency": {"limit": concurrency_limit},
            },
            "kvstore": kvstore,
        }
    else:
        spec = {
            "driver": filetype,
            "kvstore": kvstore,
        }

    if mode == "r":
        dataset_future = ts.open(spec, read=True, write=False)
    else:
        dataset_future = ts.open(spec, read=False, write=True)

    return dataset_future.result()




def _concurrency_context(concurrency_limit):
    """The per-handle context cellmap-analyze has always used: data copies and file IO
    capped, network request pools left at tensorstore's defaults."""
    if not concurrency_limit:
        return None
    return wa.capped_context(concurrency=concurrency_limit, network=False)


def _open_world_array(path, target_voxel_size, context):
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
        )
    node = wa.detect(path)
    if node.kind == "array" or (node.format == "precomputed" and node.scale_key is not None):
        return wa.open(path, context=context)
    return wa.open(path, level="first", context=context)


class ImageDataInterface:
    """cellmap-analyze's view of a dataset: worldarray for every read and for where the
    data sits, plus the integer-scaled frame (funlib ``Coordinate``/``Roi``) the
    blockwise code works in, and the zarr-python ``.ds`` handle that writes go through.

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
        from cellmap_analyze.util.io_util import (
            is_precomputed_path,
            is_remote_path,
            strip_precomputed_prefix,
        )

        dataset_path = str(dataset_path)
        if not is_remote_path(dataset_path) and not is_precomputed_path(dataset_path):
            dataset_path = str(Path(dataset_path).resolve())
        self.mode = mode
        self.concurrency_limit = concurrency_limit
        self._wa = _open_world_array(
            strip_precomputed_prefix(dataset_path),
            target_voxel_size,
            _concurrency_context(concurrency_limit),
        )
        self._wa_concurrency = concurrency_limit
        self._ds = None
        # the opened level's own path (wa.path is the path as given, e.g. the group)
        self.path = self._wa.meta.path.rstrip("/")
        self.filetype = _FILETYPES[self._wa.meta.format]
        self.dtype = self._wa.dtype
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
        if self._ds is not None:
            self._ds.voxel_size = self.voxel_size
            self._ds.roi = self.roi

    def rescale_to_factor(self, new_scale_factor):
        """Re-express the frame with another integer factor, so several datasets can
        share one frame (contact sites, masks, measure)."""
        if new_scale_factor != self.voxel_size_scale_factor:
            self._set_frame(new_scale_factor)

    @property
    def ds(self):
        """The zarr-python array (a ``CellMapArray`` in the frame), opened on first use:
        writes go through it; reads do not."""
        if self._ds is None:
            filename, dataset = split_dataset_path(self.path)
            self._ds = open_dataset(filename, dataset, mode=self.mode)
            self._ds.voxel_size = self.voxel_size
            self._ds.roi = self.roi
        return self._ds

    def __getstate__(self):
        # Workers reopen the zarr-python handle themselves on first use.
        state = self.__dict__.copy()
        state["_ds"] = None
        return state

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

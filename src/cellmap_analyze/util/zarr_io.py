import json
import logging
import os

import numpy as np
import zarr
from funlib.geometry import Coordinate, Roi

from cellmap_analyze.util.cellmap_array import CellMapArray

logger = logging.getLogger(__name__)

# Map N5 dataType strings to numpy dtypes
_N5_DTYPE_MAP = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "float32": np.float32,
    "float64": np.float64,
}


class N5ArrayMetadata:
    """Lightweight metadata wrapper for N5 arrays.

    Provides the same metadata interface as zarr.Array (shape, chunks, dtype,
    attrs) by reading the N5 attributes.json directly. Actual data reads go
    through tensorstore in ImageDataInterface.
    """

    def __init__(self, path):
        attrs_path = os.path.join(path, "attributes.json")
        with open(attrs_path) as f:
            self._attrs = json.load(f)

        self.shape = tuple(self._attrs["dimensions"])
        self.chunks = tuple(self._attrs["blockSize"])
        self.dtype = np.dtype(_N5_DTYPE_MAP[self._attrs["dataType"]])
        self.attrs = self._attrs

    def __getitem__(self, slices):
        raise NotImplementedError(
            "N5ArrayMetadata does not support direct data access. "
            "Use ImageDataInterface.to_ndarray_ts() instead."
        )

    def __setitem__(self, slices, value):
        raise NotImplementedError(
            "N5ArrayMetadata does not support direct data access."
        )


def open_dataset(filename, ds_name, mode="r"):
    """Open a zarr dataset and return a CellMapArray.

    Supports zarr v2, v3, hybrid (v2 groups with v3 arrays), and N5 formats.

    Args:
        filename: Path to the zarr container directory.
        ds_name: Name of the dataset within the container.
        mode: Open mode ('r', 'r+', 'a', 'w').

    Returns:
        A CellMapArray wrapping the dataset.
    """
    logger.debug("opening zarr dataset %s in %s", ds_name, filename)
    full_path = os.path.join(filename, ds_name)
    try:
        ds = zarr.open_array(full_path, mode=mode)
    except Exception:
        # Zarr 3.x cannot open N5 natively — fall back to reading
        # the N5 attributes.json for metadata.
        n5_attrs_path = os.path.join(full_path, "attributes.json")
        if os.path.exists(n5_attrs_path):
            logger.debug("falling back to N5 metadata reader for %s", full_path)
            ds = N5ArrayMetadata(full_path)
        else:
            logger.error("failed to open %s/%s", filename, ds_name)
            raise

    voxel_size, offset = _read_voxel_size_offset(ds)
    return CellMapArray(ds, voxel_size, offset)


def prepare_ds(
    filename,
    ds_name,
    total_roi,
    voxel_size,
    dtype,
    write_roi=None,
    write_size=None,
    num_channels=None,
    delete=False,
    force_exact_write_size=False,
    multiscales_metadata=False,
    **kwargs,
):
    """Create a zarr dataset and return a CellMapArray.

    Mirrors the old funlib.persistence.prepare_ds interface.

    Args:
        filename: Path to the zarr container directory.
        ds_name: Name of the dataset to create.
        total_roi: ROI of the dataset in world units.
        voxel_size: Size of one voxel in world units.
        dtype: Data type for the array.
        write_size: Anticipated write size in world units (determines chunk size).
        num_channels: Number of channels (prepended to shape).
        delete: Whether to overwrite existing dataset.
        force_exact_write_size: Use write_size as chunk size directly.

    Returns:
        A CellMapArray pointing to the newly created dataset.
    """
    voxel_size = Coordinate(voxel_size)

    if write_roi is not None and write_size is None:
        write_size = write_roi.shape

    if write_size is not None:
        write_size = Coordinate(write_size)
        if force_exact_write_size:
            chunk_shape = tuple(write_size / voxel_size)
        else:
            chunk_shape = tuple(write_size / voxel_size)
    else:
        chunk_shape = None

    shape = tuple(total_roi.shape / voxel_size)

    if num_channels is not None:
        shape = (num_channels,) + shape
        if chunk_shape is not None:
            chunk_shape = (num_channels,) + chunk_shape

    ds_name = ds_name.lstrip("/")

    os.makedirs(filename, exist_ok=True)

    root = zarr.open_group(filename, mode="a")
    ds = root.require_group("/".join(ds_name.split("/")[:-1])) if "/" in ds_name else root

    # Get the leaf array name
    array_name = ds_name.split("/")[-1] if "/" in ds_name else ds_name

    # Create the array, matching the container's zarr format so
    # v2 containers get v2 arrays and v3 containers get v3 arrays
    arr = zarr.open_array(
        store=os.path.join(filename, ds_name),
        shape=shape,
        chunks=chunk_shape,
        dtype=dtype,
        mode="w" if delete else "a",
        zarr_format=root.metadata.zarr_format,
    )

    # Write metadata
    arr.attrs["voxel_size"] = list(voxel_size)
    arr.attrs["offset"] = list(total_roi.begin)

    return CellMapArray(arr, voxel_size, total_roi.begin)


def _read_voxel_size_offset(ds):
    """Read voxel_size and offset from a zarr array's attributes.

    Checks multiple metadata formats: funlib-style, OME-Zarr, N5.

    Args:
        ds: A zarr.Array.

    Returns:
        (voxel_size, offset) as Coordinates.
    """
    attrs = dict(ds.attrs)

    # funlib-style: resolution/offset or voxel_size/offset
    if "resolution" in attrs:
        voxel_size = Coordinate(int(v) for v in attrs["resolution"])
    elif "voxel_size" in attrs:
        voxel_size = Coordinate(int(v) for v in attrs["voxel_size"])
    else:
        # Default to 1 for all spatial dims
        voxel_size = Coordinate(1 for _ in ds.shape)

    if "offset" in attrs:
        offset = Coordinate(int(v) for v in attrs["offset"])
    else:
        offset = Coordinate(0 for _ in voxel_size)

    return voxel_size, offset

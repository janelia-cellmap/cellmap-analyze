"""Reading arrays written the way funlib (and so cellmap-flow) writes them."""
import os

import numpy as np
import pytest
import zarr

from cellmap_analyze.util.dask_util import get_num_blocks, get_zarr_chunk_path_from_block_index
from cellmap_analyze.util.image_data_interface import ImageDataInterface


def _write(root, voxel_size, translation, offset_attr):
    """A zarr v2 OME group with one level, s0, chunked with "/" keys as funlib writes it."""
    g = zarr.open_group(str(root), mode="w", zarr_format=2)
    g.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
        "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": list(voxel_size)},
            {"type": "translation", "translation": list(translation)},
        ]}],
    }]
    a = g.create_array("s0", shape=(32, 32, 32), chunks=(8, 8, 8), dtype="uint8",
                       chunk_key_encoding={"name": "v2", "separator": "/"})
    a.attrs["voxel_size"] = list(voxel_size)
    a.attrs["offset"] = list(offset_attr)
    a[...] = np.random.default_rng(0).integers(1, 255, size=a.shape, dtype=np.uint8)
    return f"{root}/s0"


def test_funlib_corner_offset_is_accepted(tmp_path, voxel_size):
    # funlib's prepare_ds stores `offset` as voxel 0's corner, half a voxel
    # below the OME translation (its center): the same position.
    vs = [float(v) for v in voxel_size]
    translation = [1000.0 + v / 2 for v in vs]
    corner = [t - v / 2 for t, v in zip(translation, vs)]
    idi = ImageDataInterface(_write(tmp_path / "a.zarr", vs, translation, corner))
    s = idi.voxel_size_scale_factor
    assert tuple(idi.roi.offset) == tuple(int(round(c * s)) for c in corner)


def test_conflicting_offset_still_raises(tmp_path, voxel_size):
    vs = [float(v) for v in voxel_size]
    translation = [1000.0 + v / 2 for v in vs]
    offset_attr = [translation[0] + 100.0, *translation[1:]]   # neither center nor corner
    with pytest.raises(ValueError, match="Metadata conflict"):
        ImageDataInterface(_write(tmp_path / "a.zarr", vs, translation, offset_attr))


def test_zarr_v2_chunk_paths(tmp_path, voxel_size):
    vs = [float(v) for v in voxel_size]
    center = [v / 2 for v in vs]
    idi = ImageDataInterface(_write(tmp_path / "v2.zarr", vs, center, center))
    num_blocks = get_num_blocks(idi, idi.roi)
    assert num_blocks > 1
    missing = [i for i in range(num_blocks)
               if not os.path.exists(get_zarr_chunk_path_from_block_index(i, idi, depth=3))]
    assert not missing, f"no chunk file at the path given for blocks {missing[:5]}"

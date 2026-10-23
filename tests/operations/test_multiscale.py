import os

import numpy as np
import pytest
import worldarray as wa
import zarr

from cellmap_analyze.process.multiscale import Multiscale
from cellmap_analyze.util.grid_util import make_grid
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.zarr_util import create_multiscale_dataset

NOISE = 99


def _write(path, data, voxel_size, chunks, offset_voxels=(3, 5, 7)):
    """data as s0 of a cellmap-analyze group at `path`, away from the origin."""
    corner = np.multiply(offset_voxels, voxel_size)
    ds = create_multiscale_dataset(
        path, data.dtype, make_grid(data.shape, voxel_size, corner, chunks)
    )
    ds.write_index((0, 0, 0), ds.shape, data)


def _upsample_with_noise(coarse, rng):
    # Every 2x2x2 block repeats one coarse label, and in about half of them
    # one voxel becomes NOISE: 7 of 8 still agree, so mode gives back `coarse`.
    fine = coarse.repeat(2, 0).repeat(2, 1).repeat(2, 2)
    for z, y, x in np.argwhere(rng.random(coarse.shape) < 0.5):
        fine[2 * z, 2 * y, 2 * x] = NOISE
    return fine


@pytest.fixture(params=[2, 3], ids=["zarr2", "zarr3"])
def pyramid(request, tmp_path, voxel_size):
    rng = np.random.default_rng(0)
    s2 = rng.integers(0, 4, size=(4, 4, 4)).astype(np.uint64)
    s2[s2 > 0] += 2**40  # ids that need all 64 bits
    s1 = _upsample_with_noise(s2, rng)
    s0 = _upsample_with_noise(s1, rng)
    # cellmap-analyze writes arrays in the format of the container they go in
    zarr.open_group(f"{tmp_path}/ms.zarr", mode="w", zarr_format=request.param)
    path = f"{tmp_path}/ms.zarr/labels"
    _write(path, s0, voxel_size, chunks=(4, 4, 4))
    return path, [s0, s1, s2]


def test_levels_are_the_mode_of_the_level_below(pyramid):
    path, expected = pyramid
    Multiscale(path, num_workers=1).create_multiscale()

    # 16^3 in 4^3 chunks: s1 is 8^3, s2 is 4^3 and fits one chunk, so it is the last.
    assert not os.path.exists(f"{path}/s3")
    for level in (1, 2):
        data = ImageDataInterface(f"{path}/s{level}").to_ndarray_ts()
        assert np.array_equal(data, expected[level])


def test_levels_cover_the_same_physical_box(pyramid, voxel_size):
    path, _ = pyramid
    Multiscale(path, num_workers=1).create_multiscale()

    # placements as the group's multiscales document lists them
    placements = wa.open_pyramid(path).placements()
    assert [name for name, _, _ in placements] == ["s0", "s1", "s2"]
    _, scale0, translation0 = (np.array(p) for p in placements[0])
    for level, (_, scale, translation) in enumerate(placements):
        factor = 2**level
        assert np.allclose(scale, scale0 * factor)
        # voxel 0 of level k is centred on the 2^k finer voxels it summarises
        assert np.allclose(translation, translation0 + scale0 * (factor - 1) / 2)

    # Read back, every level starts and ends where s0 does.
    boxes = [ImageDataInterface(f"{path}/s{level}").grid.box for level in range(3)]
    for box in boxes:
        assert np.allclose(box.begin, boxes[0].begin) and np.allclose(box.end, boxes[0].end)


def test_partial_edge_blocks_are_kept(tmp_path, voxel_size):
    data = np.zeros((5, 4, 4), np.uint8)
    data[4] = 3  # a last plane with no partner at s1
    path = f"{tmp_path}/ms.zarr/odd"
    _write(path, data, voxel_size, chunks=(2, 2, 2))
    Multiscale(path, num_workers=1).create_multiscale()

    s1 = ImageDataInterface(f"{path}/s1").to_ndarray_ts()
    assert s1.shape == (3, 2, 2)
    assert np.all(s1[2] == 3) and not np.any(s1[:2])


def test_floats_default_to_mean(tmp_path, voxel_size):
    data = np.zeros((4, 4, 4), np.float32)
    data[:2, :2, :2] = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    path = f"{tmp_path}/ms.zarr/edt"
    _write(path, data, voxel_size, chunks=(2, 2, 2))
    Multiscale(path, num_workers=1, num_levels=1).create_multiscale()

    s1 = ImageDataInterface(f"{path}/s1").to_ndarray_ts()
    assert s1[0, 0, 0] == 3.5 and not np.any(s1.ravel()[1:])


def test_commands_pyramid_their_output_and_filled_copy(tmp_path, voxel_size):
    from cli.cli import RunProperties

    data = np.ones((8, 8, 8), np.uint8)
    for name in ("seg", "seg_filled"):
        _write(f"{tmp_path}/out.zarr/{name}", data, voxel_size, chunks=(4, 4, 4))
    rp = RunProperties.__new__(RunProperties)  # without parsing a command line
    rp.multiscale = True
    rp.add_multiscale(f"{tmp_path}/out.zarr/seg", 1, filled=True)
    assert os.path.exists(f"{tmp_path}/out.zarr/seg/s1")
    assert os.path.exists(f"{tmp_path}/out.zarr/seg_filled/s1")
    # an output the command did not write is skipped
    rp.add_multiscale(f"{tmp_path}/out.zarr/missing", 1)


def test_anisotropic_axes_wait_for_the_finer_ones(tmp_path):
    path = f"{tmp_path}/ms.zarr/aniso"
    _write(path, np.zeros((16, 16, 16), np.uint8), (32, 8, 8), chunks=(2, 2, 2))
    factors = [level["factors"] for level in Multiscale(path, num_levels=3).levels]
    assert factors == [(1, 2, 2), (1, 2, 2), (2, 2, 2)]


def test_existing_levels_need_overwrite(pyramid):
    path, expected = pyramid
    Multiscale(path, num_workers=1, num_levels=1).create_multiscale()
    with pytest.raises(FileExistsError):
        Multiscale(path, num_workers=1)
    Multiscale(path, num_workers=1, overwrite=True).create_multiscale()
    assert np.array_equal(ImageDataInterface(f"{path}/s2").to_ndarray_ts(), expected[2])

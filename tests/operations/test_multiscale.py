import os

import numpy as np
import pytest
import zarr
from funlib.geometry import Coordinate, Roi

from cellmap_analyze.process.multiscale import Multiscale, downsample_factors
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.voxel_size_utils import (
    compute_common_scale_factor,
    get_multiscales,
    scale_voxel_size_to_integers,
)
from cellmap_analyze.util.zarr_util import create_multiscale_dataset

NOISE = 99


def _write(path, data, voxel_size, chunks, offset_voxels=(3, 5, 7)):
    """data as s0 of a cellmap-analyze group at `path`, away from the origin."""
    original = tuple(float(v) for v in voxel_size)
    scaled, _ = scale_voxel_size_to_integers(original)
    roi = Roi(
        Coordinate(offset_voxels) * Coordinate(scaled),
        Coordinate(data.shape) * Coordinate(scaled),
    )
    ds = create_multiscale_dataset(
        path,
        dtype=data.dtype,
        voxel_size=scaled,
        total_roi=roi,
        write_size=Coordinate(chunks) * Coordinate(scaled),
        original_voxel_size=original,
    )
    ds.data[:] = data


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

    datasets = get_multiscales(zarr.open_group(path, mode="r").attrs.asdict())[0]["datasets"]
    assert [d["path"] for d in datasets] == ["s0", "s1", "s2"]
    transforms = [
        [np.array(t.get("scale", t.get("translation"))) for t in d["coordinateTransformations"]]
        for d in datasets
    ]
    scale0, translation0 = transforms[0]
    for level, (scale, translation) in enumerate(transforms):
        factor = 2**level
        assert np.allclose(scale, scale0 * factor)
        # voxel 0 of level k is centred on the 2^k finer voxels it summarises
        assert np.allclose(translation, translation0 + scale0 * (factor - 1) / 2)

    # Read back, every level starts and ends where s0 does.
    idis = [ImageDataInterface(f"{path}/s{level}") for level in range(3)]
    common = compute_common_scale_factor(*(idi.voxel_size_scale_factor for idi in idis))
    for idi in idis:
        idi.rescale_to_factor(common)
    assert all(idi.roi == idis[0].roi for idi in idis)


def test_partial_edge_blocks_are_kept(tmp_path, voxel_size):
    data = np.zeros((5, 4, 4), np.uint8)
    data[4] = 3  # a last plane with no partner at s1
    path = f"{tmp_path}/ms.zarr/odd"
    _write(path, data, voxel_size, chunks=(2, 2, 2))
    Multiscale(path, num_workers=1).create_multiscale()

    s1 = ImageDataInterface(f"{path}/s1").to_ndarray_ts()
    assert s1.shape == (3, 2, 2)
    assert np.all(s1[2] == 3) and not np.any(s1[:2])


def test_anisotropic_axes_wait_for_the_finer_ones():
    assert downsample_factors((8, 8, 20)) == (2, 2, 1)
    assert downsample_factors((16, 16, 20)) == (2, 2, 2)
    assert downsample_factors((32, 8, 8)) == (1, 2, 2)
    assert downsample_factors((3.54, 4, 4.5)) == (2, 2, 2)


def test_existing_levels_need_overwrite(pyramid):
    path, expected = pyramid
    Multiscale(path, num_workers=1, num_levels=1).create_multiscale()
    with pytest.raises(FileExistsError):
        Multiscale(path, num_workers=1)
    Multiscale(path, num_workers=1, overwrite=True).create_multiscale()
    assert np.array_equal(ImageDataInterface(f"{path}/s2").to_ndarray_ts(), expected[2])

"""Arrays stored x, y, z (as TensorSwitch writes OME-Zarr v3) are read z, y, x."""

import numpy as np
import pandas as pd
import worldarray as wa
from pandas.testing import assert_frame_equal

from cellmap_analyze.analyze.measure import Measure
from cellmap_analyze.process.contact_sites import ContactSites
from cellmap_analyze.process.multiscale import Multiscale
from cellmap_analyze.util.grid_util import make_grid
from cellmap_analyze.util.zarr_util import create_multiscale_dataset, open_dataset
from tests.test_utils import read_all

CHUNKS = (4, 4, 4)


def _write_xyz(path, data, voxel_size, corner=(0, 0, 0)):
    """``data`` (z, y, x) as s0 of an OME-Zarr v3 group at ``path`` stored x, y, z."""
    voxel_size = np.asarray(voxel_size, dtype=float)
    translation = np.asarray(corner, dtype=float) + voxel_size / 2
    vol = wa.create(
        f"{path}/s0",
        data.shape[::-1],
        data.dtype,
        tuple(voxel_size[::-1]),
        tuple(translation[::-1]),
        axes=("x", "y", "z"),
        chunks=CHUNKS[::-1],
    )
    vol.write_index((0, 0, 0), vol.shape, data.transpose(2, 1, 0))
    return f"{path}/s0"


def _write_zyx(path, data, voxel_size, corner=(0, 0, 0)):
    ds = create_multiscale_dataset(
        path, data.dtype, make_grid(data.shape, voxel_size, corner, chunks=CHUNKS)
    )
    ds.write_index((0, 0, 0), ds.shape, data)
    return f"{path}/s0"


def test_an_xyz_store_is_read_zyx(tmp_path, segmentation_1, voxel_size):
    corner = np.array([3, 5, 7]) * voxel_size
    path = _write_xyz(f"{tmp_path}/xyz.zarr/seg", segmentation_1, voxel_size, corner)
    assert wa.open(path).axes == ("x", "y", "z")

    vol = open_dataset(path)
    assert vol.axes == ("z", "y", "x")
    assert vol.shape == segmentation_1.shape
    np.testing.assert_allclose(vol.voxel_size, voxel_size)
    np.testing.assert_allclose(vol.corner, corner)
    assert np.array_equal(read_all(path), segmentation_1)


def test_measure_reads_an_xyz_store_as_its_zyx_twin(
    tmp_path, tmp_zarr, segmentation_1, voxel_size
):
    xyz = _write_xyz(f"{tmp_path}/xyz.zarr/segmentation_1", segmentation_1, voxel_size)
    for name, path in (("zyx", f"{tmp_zarr}/segmentation_1/s0"), ("xyz", xyz)):
        Measure(
            input_path=path, output_path=f"{tmp_path}/{name}_csvs", num_workers=1
        ).get_measurements()
    # COM Z, Y and X are the store's z, y and x, not its first, second and third axes
    assert_frame_equal(
        pd.read_csv(f"{tmp_path}/xyz_csvs/segmentation_1.csv"),
        pd.read_csv(f"{tmp_path}/zyx_csvs/segmentation_1.csv"),
    )


def test_contact_sites_between_an_xyz_and_a_zyx_store(
    tmp_path, tmp_zarr, segmentation_2, voxel_size
):
    # Both are read onto one z, y, x grid; before, the x, y, z one raised
    xyz = _write_xyz(f"{tmp_path}/xyz.zarr/segmentation_2", segmentation_2, voxel_size)
    for name, organelle_2 in (("zyx", f"{tmp_zarr}/segmentation_2/s0"), ("xyz", xyz)):
        ContactSites(
            f"{tmp_zarr}/segmentation_1/s0",
            organelle_2,
            f"{tmp_path}/{name}_out.zarr/contact_sites",
            contact_distance_nm=8,
            minimum_volume_nm_3=0,
            num_workers=1,
        ).get_contact_sites()

    expected = read_all(f"{tmp_path}/zyx_out.zarr/contact_sites/s0")
    assert np.any(expected)
    assert np.array_equal(read_all(f"{tmp_path}/xyz_out.zarr/contact_sites/s0"), expected)
    assert open_dataset(f"{tmp_path}/xyz_out.zarr/contact_sites/s0").axes == ("z", "y", "x")


def test_multiscale_adds_xyz_levels_to_an_xyz_store(tmp_path, voxel_size):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4, size=(16, 12, 8)).astype(np.uint64)
    corner = np.array([3, 5, 7]) * voxel_size
    zyx = _write_zyx(f"{tmp_path}/zyx.zarr/labels", data, voxel_size, corner)
    xyz = _write_xyz(f"{tmp_path}/xyz.zarr/labels", data, voxel_size, corner)
    for path in (zyx, xyz):
        Multiscale(path, num_workers=1).create_multiscale()

    zyx_group, xyz_group = zyx.rsplit("/", 1)[0], xyz.rsplit("/", 1)[0]
    for level in ("s1", "s2"):
        assert wa.open(f"{xyz_group}/{level}").axes == ("x", "y", "z")
        got = open_dataset(f"{xyz_group}/{level}")
        expected = open_dataset(f"{zyx_group}/{level}")
        assert got.shape == expected.shape
        np.testing.assert_allclose(got.voxel_size, expected.voxel_size)
        np.testing.assert_allclose(got.corner, expected.corner)
        assert np.array_equal(
            read_all(f"{xyz_group}/{level}"), read_all(f"{zyx_group}/{level}")
        )

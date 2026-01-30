import pytest
from cellmap_analyze.process.contact_sites import ContactSites
import numpy as np

from cellmap_analyze.util.xarray_image_data_interface import XarrayImageDataInterface


def test_contact_site_whole_1(segmentation_1, segmentation_2, contact_sites_distance_1):
    cs = ContactSites.get_ndarray_contact_sites(
        segmentation_1, segmentation_2, 1, zero_pad=True
    )
    assert np.array_equal(cs, contact_sites_distance_1)


def test_contact_site_whole_2(segmentation_1, segmentation_2, contact_sites_distance_2):
    cs = ContactSites.get_ndarray_contact_sites(
        segmentation_1, segmentation_2, 2, zero_pad=True
    )
    assert np.array_equal(cs, contact_sites_distance_2)


def test_contact_site_whole_3(segmentation_1, segmentation_2, contact_sites_distance_3):
    cs = ContactSites.get_ndarray_contact_sites(
        segmentation_1, segmentation_2, 3, zero_pad=True
    )
    assert np.array_equal(cs, contact_sites_distance_3)


@pytest.mark.parametrize("contact_distance", [1, 2, 3])
def test_contact_site_blocks(tmp_zarr, voxel_size, contact_distance):
    cs = ContactSites(
        f"{tmp_zarr}/segmentation_1/s0",
        f"{tmp_zarr}/segmentation_2/s0",
        tmp_zarr + f"/test_contact_sites_distance_{contact_distance}",
        voxel_size * contact_distance,
        minimum_volume_nm_3=0,
        num_workers=1,
    )
    cs.get_contact_sites()

    ground_truth = XarrayImageDataInterface(
        f"{tmp_zarr}/contact_sites_distance_{contact_distance}/s0"
    ).to_ndarray_ts()
    test_data = XarrayImageDataInterface(
        f"{tmp_zarr}/test_contact_sites_distance_{contact_distance}/s0"
    ).to_ndarray_ts()
    assert np.array_equal(
        test_data,
        ground_truth,
    )


@pytest.mark.parametrize("contact_distance", [1, 2, 3])
def test_different_voxel_sizes(tmp_zarr, voxel_size, contact_distance):

    cs = ContactSites(
        f"{tmp_zarr}/segmentation_1_downsampled/s0",
        f"{tmp_zarr}/segmentation_2/s0",
        tmp_zarr + f"/test_downsampled_contact_sites_distance_{contact_distance}",
        voxel_size * contact_distance,
        minimum_volume_nm_3=0,
        num_workers=1,
    )
    cs.get_contact_sites()

    ground_truth = XarrayImageDataInterface(
        f"{tmp_zarr}/contact_sites_distance_{contact_distance}/s0"
    ).to_ndarray_ts()
    test_data = XarrayImageDataInterface(
        f"{tmp_zarr}/test_downsampled_contact_sites_distance_{contact_distance}/s0"
    ).to_ndarray_ts()

    assert np.array_equal(
        test_data,
        ground_truth,
    )

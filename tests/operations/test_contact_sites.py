import pytest
from cellmap_analyze.process.contact_sites import ContactSites
import numpy as np

from cellmap_analyze.util.image_data_interface import (
    open_ds_tensorstore,
    to_ndarray_tensorstore,
)


def test_contact_site_whole_1(segmentation_1, segmentation_2, contact_sites_distance_1):
    cs = ContactSites.get_ndarray_contact_sites(segmentation_1, segmentation_2, 1)
    assert np.array_equal(cs, contact_sites_distance_1)


def test_contact_site_whole_2(segmentation_1, segmentation_2, contact_sites_distance_2):
    cs = ContactSites.get_ndarray_contact_sites(segmentation_1, segmentation_2, 2)
    assert np.array_equal(cs, contact_sites_distance_2)


def test_contact_site_whole_3(segmentation_1, segmentation_2, contact_sites_distance_3):
    cs = ContactSites.get_ndarray_contact_sites(segmentation_1, segmentation_2, 3)
    assert np.array_equal(cs, contact_sites_distance_3)


@pytest.mark.parametrize("contact_distance", [1, 2, 3])
def test_contact_site_blocks(tmp_zarr, voxel_size, contact_distance):
    segmentation_1_path = f"{tmp_zarr}/segmentation_1/s0"
    segmentation_2_path = f"{tmp_zarr}/segmentation_2/s0"

    cs = ContactSites(
        segmentation_1_path,
        segmentation_2_path,
        tmp_zarr + f"/test_contact_sites_distance_{contact_distance}",
        voxel_size * contact_distance,
        minimum_volume_nm_3=0,
        num_workers=1,
    )
    cs.get_contact_sites()

    ground_truth_path = f"{tmp_zarr}/contact_sites_distance_{contact_distance}/s0"
    ground_truth = to_ndarray_tensorstore(open_ds_tensorstore(ground_truth_path))

    test_path = f"{tmp_zarr}/test_contact_sites_distance_{contact_distance}/s0"
    test_data = to_ndarray_tensorstore(open_ds_tensorstore(test_path))

    assert np.array_equal(
        test_data,
        ground_truth,
    )

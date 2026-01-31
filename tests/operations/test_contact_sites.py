import pytest
from cellmap_analyze.process.contact_sites import ContactSites
import numpy as np

from cellmap_analyze.util.image_data_interface import ImageDataInterface


def test_contact_site_whole_1(segmentation_1, segmentation_2, contact_sites_distance_1):
    # Test using legacy behavior (no voxel_size/contact_distance_nm parameters)
    # For isotropic data only - anisotropic tests will fail with this approach
    cs = ContactSites.get_ndarray_contact_sites(
        segmentation_1, segmentation_2, 1, zero_pad=True
    )
    assert np.array_equal(cs, contact_sites_distance_1)


def test_contact_site_whole_2(
    segmentation_1, segmentation_2, contact_sites_distance_2, voxel_size
):
    # Legacy behavior (no voxel_size/contact_distance_nm) - isotropic only
    if not np.all(voxel_size == voxel_size[0]):
        pytest.skip("Legacy contact site API is isotropic only")
    cs = ContactSites.get_ndarray_contact_sites(
        segmentation_1, segmentation_2, 2, zero_pad=True
    )
    assert np.array_equal(cs, contact_sites_distance_2)


def test_contact_site_whole_3(
    segmentation_1, segmentation_2, contact_sites_distance_3, voxel_size
):
    # Legacy behavior (no voxel_size/contact_distance_nm) - isotropic only
    if not np.all(voxel_size == voxel_size[0]):
        pytest.skip("Legacy contact site API is isotropic only")
    cs = ContactSites.get_ndarray_contact_sites(
        segmentation_1, segmentation_2, 3, zero_pad=True
    )
    assert np.array_equal(cs, contact_sites_distance_3)


@pytest.mark.parametrize("contact_distance", [1, 2, 3])
def test_contact_site_blocks(tmp_zarr, voxel_size, contact_distance):
    # For anisotropic data, use minimum voxel size to convert distance to nm
    if np.isscalar(voxel_size):
        contact_distance_nm = voxel_size * contact_distance
    else:
        contact_distance_nm = float(min(voxel_size)) * contact_distance

    cs = ContactSites(
        f"{tmp_zarr}/segmentation_1/s0",
        f"{tmp_zarr}/segmentation_2/s0",
        tmp_zarr + f"/test_contact_sites_distance_{contact_distance}",
        contact_distance_nm,
        minimum_volume_nm_3=0,
        num_workers=1,
    )
    cs.get_contact_sites()

    ground_truth = ImageDataInterface(
        f"{tmp_zarr}/contact_sites_distance_{contact_distance}/s0"
    ).to_ndarray_ts()
    test_data = ImageDataInterface(
        f"{tmp_zarr}/test_contact_sites_distance_{contact_distance}/s0"
    ).to_ndarray_ts()

    # Save test results for debugging
    if not np.array_equal(test_data, ground_truth):
        import os
        debug_dir = "/tmp/contact_sites_debug"
        os.makedirs(debug_dir, exist_ok=True)
        voxel_str = f"{voxel_size[0]}_{voxel_size[1]}_{voxel_size[2]}" if not np.isscalar(voxel_size) else f"{voxel_size}"
        np.save(f"{debug_dir}/test_data_{voxel_str}_dist{contact_distance}.npy", test_data)
        np.save(f"{debug_dir}/ground_truth_{voxel_str}_dist{contact_distance}.npy", ground_truth)
        print(f"\nSaved debug arrays to {debug_dir}/")
        print(f"Test data shape: {test_data.shape}, unique values: {np.unique(test_data)}")
        print(f"Ground truth shape: {ground_truth.shape}, unique values: {np.unique(ground_truth)}")
        print(f"Diff: {np.sum(test_data != ground_truth)} voxels differ")

    assert np.array_equal(
        test_data,
        ground_truth,
    )


@pytest.mark.parametrize("contact_distance", [1, 2, 3])
def test_different_voxel_sizes(tmp_zarr, voxel_size, contact_distance):
    # For anisotropic data, use minimum voxel size to convert distance to nm
    if np.isscalar(voxel_size):
        contact_distance_nm = voxel_size * contact_distance
    else:
        contact_distance_nm = float(min(voxel_size)) * contact_distance

    cs = ContactSites(
        f"{tmp_zarr}/segmentation_1_downsampled/s0",
        f"{tmp_zarr}/segmentation_2/s0",
        tmp_zarr + f"/test_downsampled_contact_sites_distance_{contact_distance}",
        contact_distance_nm,
        minimum_volume_nm_3=0,
        num_workers=1,
    )
    cs.get_contact_sites()

    ground_truth = ImageDataInterface(
        f"{tmp_zarr}/contact_sites_distance_{contact_distance}/s0"
    ).to_ndarray_ts()
    test_data = ImageDataInterface(
        f"{tmp_zarr}/test_downsampled_contact_sites_distance_{contact_distance}/s0"
    ).to_ndarray_ts()

    # Save test results for debugging
    if not np.array_equal(test_data, ground_truth):
        import os
        debug_dir = "/tmp/contact_sites_debug"
        os.makedirs(debug_dir, exist_ok=True)
        voxel_str = f"{voxel_size[0]}_{voxel_size[1]}_{voxel_size[2]}" if not np.isscalar(voxel_size) else f"{voxel_size}"
        np.save(f"{debug_dir}/test_data_downsampled_{voxel_str}_dist{contact_distance}.npy", test_data)
        np.save(f"{debug_dir}/ground_truth_downsampled_{voxel_str}_dist{contact_distance}.npy", ground_truth)
        print(f"\nSaved debug arrays to {debug_dir}/")
        print(f"Test data shape: {test_data.shape}, unique values: {np.unique(test_data)}")
        print(f"Ground truth shape: {ground_truth.shape}, unique values: {np.unique(ground_truth)}")
        print(f"Diff: {np.sum(test_data != ground_truth)} voxels differ")

    assert np.array_equal(
        test_data,
        ground_truth,
    )

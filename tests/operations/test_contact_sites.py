import pytest
from cellmap_analyze.process.contact_sites import ContactSites
import numpy as np
from funlib.geometry import Roi, Coordinate

from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.zarr_util import create_multiscale_dataset
from tests.contact_site_fixture_helper import compute_contact_sites_ground_truth


def test_contact_site_whole_8nm(
    segmentation_1, segmentation_2, contact_sites_distance_8nm, voxel_size
):
    contact_distance_nm = 8
    contact_distance_voxels = contact_distance_nm / float(np.min(voxel_size))
    cs = ContactSites.get_ndarray_contact_sites(
        segmentation_1, segmentation_2, contact_distance_voxels, zero_pad=True,
        voxel_size=voxel_size, contact_distance_nm=contact_distance_nm,
    )
    assert np.array_equal(cs, contact_sites_distance_8nm)


def test_contact_site_whole_16nm(
    segmentation_1, segmentation_2, contact_sites_distance_16nm, voxel_size
):
    contact_distance_nm = 16
    contact_distance_voxels = contact_distance_nm / float(np.min(voxel_size))
    cs = ContactSites.get_ndarray_contact_sites(
        segmentation_1, segmentation_2, contact_distance_voxels, zero_pad=True,
        voxel_size=voxel_size, contact_distance_nm=contact_distance_nm,
    )
    assert np.array_equal(cs, contact_sites_distance_16nm)


def test_contact_site_whole_24nm(
    segmentation_1, segmentation_2, contact_sites_distance_24nm, voxel_size
):
    contact_distance_nm = 24
    contact_distance_voxels = contact_distance_nm / float(np.min(voxel_size))
    cs = ContactSites.get_ndarray_contact_sites(
        segmentation_1, segmentation_2, contact_distance_voxels, zero_pad=True,
        voxel_size=voxel_size, contact_distance_nm=contact_distance_nm,
    )
    assert np.array_equal(cs, contact_sites_distance_24nm)


@pytest.mark.parametrize("contact_distance_nm", [8, 16, 24])
def test_contact_site_blocks(tmp_zarr, voxel_size, contact_distance_nm):
    cs = ContactSites(
        f"{tmp_zarr}/segmentation_1/s0",
        f"{tmp_zarr}/segmentation_2/s0",
        tmp_zarr + f"/test_contact_sites_distance_{contact_distance_nm}nm",
        contact_distance_nm,
        minimum_volume_nm_3=0,
        num_workers=1,
    )
    cs.get_contact_sites()

    ground_truth = ImageDataInterface(
        f"{tmp_zarr}/contact_sites_distance_{contact_distance_nm}nm/s0"
    ).to_ndarray_ts()
    test_data = ImageDataInterface(
        f"{tmp_zarr}/test_contact_sites_distance_{contact_distance_nm}nm/s0"
    ).to_ndarray_ts()

    # Save test results for debugging
    if not np.array_equal(test_data, ground_truth):
        import os
        debug_dir = "/tmp/contact_sites_debug"
        os.makedirs(debug_dir, exist_ok=True)
        voxel_str = f"{voxel_size[0]}_{voxel_size[1]}_{voxel_size[2]}" if not np.isscalar(voxel_size) else f"{voxel_size}"
        np.save(f"{debug_dir}/test_data_{voxel_str}_dist{contact_distance_nm}nm.npy", test_data)
        np.save(f"{debug_dir}/ground_truth_{voxel_str}_dist{contact_distance_nm}nm.npy", ground_truth)
        print(f"\nSaved debug arrays to {debug_dir}/")
        print(f"Test data shape: {test_data.shape}, unique values: {np.unique(test_data)}")
        print(f"Ground truth shape: {ground_truth.shape}, unique values: {np.unique(ground_truth)}")
        print(f"Diff: {np.sum(test_data != ground_truth)} voxels differ")

    assert np.array_equal(
        test_data,
        ground_truth,
    )


@pytest.mark.parametrize("contact_distance_nm", [8, 16, 24])
def test_contact_site_blocks_mismatched_voxel_sizes(tmp_zarr, voxel_size, contact_distance_nm):
    cs = ContactSites(
        f"{tmp_zarr}/segmentation_1_downsampled/s0",
        f"{tmp_zarr}/segmentation_2/s0",
        tmp_zarr + f"/test_downsampled_contact_sites_distance_{contact_distance_nm}nm",
        contact_distance_nm,
        minimum_volume_nm_3=0,
        num_workers=1,
    )
    cs.get_contact_sites()

    ground_truth = ImageDataInterface(
        f"{tmp_zarr}/contact_sites_distance_{contact_distance_nm}nm/s0"
    ).to_ndarray_ts()
    test_data = ImageDataInterface(
        f"{tmp_zarr}/test_downsampled_contact_sites_distance_{contact_distance_nm}nm/s0"
    ).to_ndarray_ts()

    # Save test results for debugging
    if not np.array_equal(test_data, ground_truth):
        import os
        debug_dir = "/tmp/contact_sites_debug"
        os.makedirs(debug_dir, exist_ok=True)
        voxel_str = f"{voxel_size[0]}_{voxel_size[1]}_{voxel_size[2]}" if not np.isscalar(voxel_size) else f"{voxel_size}"
        np.save(f"{debug_dir}/test_data_downsampled_{voxel_str}_dist{contact_distance_nm}nm.npy", test_data)
        np.save(f"{debug_dir}/ground_truth_downsampled_{voxel_str}_dist{contact_distance_nm}nm.npy", ground_truth)
        print(f"\nSaved debug arrays to {debug_dir}/")
        print(f"Test data shape: {test_data.shape}, unique values: {np.unique(test_data)}")
        print(f"Ground truth shape: {ground_truth.shape}, unique values: {np.unique(ground_truth)}")
        print(f"Diff: {np.sum(test_data != ground_truth)} voxels differ")

    assert np.array_equal(
        test_data,
        ground_truth,
    )


@pytest.mark.parametrize("contact_distance_nm", [8, 16, 24])
def test_contact_site_blocks_different_anisotropic_voxel_sizes(
    tmp_path, contact_distance_nm
):
    """Test contact sites between two datasets with genuinely different anisotropic voxel sizes.

    Unlike test_contact_site_blocks_mismatched_voxel_sizes which uses a scalar 2x multiple,
    this test uses voxel sizes that differ independently per axis (e.g. (4,6,8) vs (6,4,6)),
    resulting in non-integer rescale factors when resampling to the element-wise minimum.
    """
    voxel_size_1 = (4, 6, 8)
    voxel_size_2 = (6, 4, 6)
    output_voxel_size = tuple(
        min(v1, v2) for v1, v2 in zip(voxel_size_1, voxel_size_2)
    )  # (4, 4, 6)

    # Create segmentation arrays at their native resolutions
    # Physical ROI must be a multiple of both voxel sizes
    # LCM-friendly physical size: 120nm per axis (divisible by 4,6,8)
    physical_size = 120
    shape_1 = tuple(physical_size // v for v in voxel_size_1)  # (30, 20, 15)
    shape_2 = tuple(physical_size // v for v in voxel_size_2)  # (20, 30, 20)

    seg_1 = np.zeros(shape_1, dtype=np.uint8)
    seg_1[2:6, 2:6, 2:6] = 1

    seg_2 = np.zeros(shape_2, dtype=np.uint8)
    seg_2[3:8, 5:10, 3:8] = 1

    # Write to zarr with their respective voxel sizes
    zarr_path = str(tmp_path / "test.zarr")
    chunk_size = (4, 4, 4)

    for name, seg, vs in [
        ("seg_1", seg_1, voxel_size_1),
        ("seg_2", seg_2, voxel_size_2),
    ]:
        data_path = f"{zarr_path}/{name}"
        total_roi = Roi((0, 0, 0), tuple(s * v for s, v in zip(seg.shape, vs)))
        write_size = tuple(c * v for c, v in zip(chunk_size, vs))
        ds = create_multiscale_dataset(
            data_path,
            dtype=seg.dtype,
            voxel_size=vs,
            total_roi=total_roi,
            write_size=write_size,
        )
        ds.data[:] = seg

    # Run ContactSites blockwise
    output_path = f"{zarr_path}/contact_sites_{contact_distance_nm}nm"
    cs = ContactSites(
        f"{zarr_path}/seg_1/s0",
        f"{zarr_path}/seg_2/s0",
        output_path,
        contact_distance_nm,
        minimum_volume_nm_3=0,
        num_workers=1,
    )
    cs.get_contact_sites()

    test_data = ImageDataInterface(f"{output_path}/s0").to_ndarray_ts()

    # Compute ground truth: read both arrays resampled to output_voxel_size
    # over the same ROI, then run ndarray contact sites
    idi_1 = ImageDataInterface(
        f"{zarr_path}/seg_1/s0", output_voxel_size=Coordinate(output_voxel_size)
    )
    idi_2 = ImageDataInterface(
        f"{zarr_path}/seg_2/s0", output_voxel_size=Coordinate(output_voxel_size)
    )
    roi = idi_1.roi.intersect(idi_2.roi)
    # Snap ROI to output voxel size grid to ensure consistent shapes
    ovs = Coordinate(output_voxel_size)
    snapped_roi = Roi(
        roi.offset,
        Coordinate(
            int(s // v) * v for s, v in zip(roi.shape, ovs)
        ),
    )
    resampled_1 = idi_1.to_ndarray_ts(snapped_roi)
    resampled_2 = idi_2.to_ndarray_ts(snapped_roi)

    # Crop to matching shape in case of rounding differences
    min_shape = tuple(min(s1, s2) for s1, s2 in zip(resampled_1.shape, resampled_2.shape))
    resampled_1 = resampled_1[:min_shape[0], :min_shape[1], :min_shape[2]]
    resampled_2 = resampled_2[:min_shape[0], :min_shape[1], :min_shape[2]]

    output_voxel_size_arr = np.array(output_voxel_size)
    ground_truth = compute_contact_sites_ground_truth(
        resampled_1, resampled_2, contact_distance_nm, output_voxel_size_arr
    )

    # Crop test_data to same shape (blockwise output may have different extent)
    test_data = test_data[:min_shape[0], :min_shape[1], :min_shape[2]]

    # Contact site locations should match (IDs may differ due to blockwise labeling)
    assert np.array_equal(test_data > 0, ground_truth > 0)

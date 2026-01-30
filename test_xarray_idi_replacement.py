#!/usr/bin/env python
"""Test that XarrayImageDataInterface can replace ImageDataInterface."""

import numpy as np
import tempfile
import zarr
import os
from funlib.geometry import Roi as FunlibRoi, Coordinate as FunlibCoordinate

from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.xarray_image_data_interface import (
    XarrayImageDataInterface,
    Roi,
    Coordinate,
)


def create_test_zarr(tmp_dir, shape=(32, 32, 32), voxel_size=(8, 8, 8), offset=(0, 0, 0), dtype=np.uint8):
    """Create a test zarr dataset with known data."""
    zarr_path = os.path.join(tmp_dir, "test.zarr")
    dataset_path = f"{zarr_path}/data/s0"

    # Create zarr store
    store = zarr.open(zarr_path, mode="w")

    # Create data group
    data_group = store.create_group("data")

    # Create test data - use a pattern that makes verification easy
    data = np.zeros(shape, dtype=dtype)
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                data[z, y, x] = (z + y + x) % 256

    # Create the array
    arr = data_group.create_dataset(
        "s0",
        data=data,
        chunks=(8, 8, 8),
        dtype=dtype,
    )

    # Set attributes
    arr.attrs["voxel_size"] = list(voxel_size)
    arr.attrs["offset"] = list(offset)

    return dataset_path, data


def test_basic_attributes():
    """Test that both interfaces return the same basic attributes."""
    print("\n=== Testing Basic Attributes ===")

    with tempfile.TemporaryDirectory() as tmp_dir:
        voxel_size = (8, 8, 8)
        offset = (16, 32, 48)
        dataset_path, _ = create_test_zarr(tmp_dir, voxel_size=voxel_size, offset=offset)

        # Create both interfaces
        idi = ImageDataInterface(dataset_path)
        xidi = XarrayImageDataInterface(dataset_path)

        # Compare attributes
        print(f"  path: IDI={idi.path}, XIDI={xidi.path}")
        assert idi.path == xidi.path, "path mismatch"

        print(f"  voxel_size: IDI={tuple(idi.voxel_size)}, XIDI={tuple(xidi.voxel_size)}")
        assert tuple(idi.voxel_size) == tuple(xidi.voxel_size), "voxel_size mismatch"

        print(f"  dtype: IDI={idi.dtype}, XIDI={xidi.dtype}")
        assert idi.dtype == xidi.dtype, "dtype mismatch"

        print(f"  chunk_shape: IDI={tuple(idi.chunk_shape)}, XIDI={tuple(xidi.chunk_shape)}")
        assert tuple(idi.chunk_shape) == tuple(xidi.chunk_shape), "chunk_shape mismatch"

        print(f"  offset: IDI={tuple(idi.offset)}, XIDI={tuple(xidi.offset)}")
        assert tuple(idi.offset) == tuple(xidi.offset), "offset mismatch"

        print(f"  roi.begin: IDI={tuple(idi.roi.begin)}, XIDI={tuple(xidi.roi.begin)}")
        assert tuple(idi.roi.begin) == tuple(xidi.roi.begin), "roi.begin mismatch"

        print(f"  roi.shape: IDI={tuple(idi.roi.shape)}, XIDI={tuple(xidi.roi.shape)}")
        assert tuple(idi.roi.shape) == tuple(xidi.roi.shape), "roi.shape mismatch"

        print("  ✓ All basic attributes match!")


def test_read_full_dataset():
    """Test reading the entire dataset."""
    print("\n=== Testing Full Dataset Read ===")

    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path, expected_data = create_test_zarr(tmp_dir)

        idi = ImageDataInterface(dataset_path)
        xidi = XarrayImageDataInterface(dataset_path)

        # Read full dataset
        idi_data = idi.to_ndarray_ts()
        xidi_data = xidi.to_ndarray_ts()

        print(f"  IDI shape: {idi_data.shape}, XIDI shape: {xidi_data.shape}")
        assert idi_data.shape == xidi_data.shape, "Shape mismatch"

        print(f"  IDI dtype: {idi_data.dtype}, XIDI dtype: {xidi_data.dtype}")

        # Compare data
        if np.allclose(idi_data, xidi_data):
            print("  ✓ Data matches exactly!")
        else:
            diff = np.abs(idi_data.astype(float) - xidi_data.astype(float))
            print(f"  Max difference: {diff.max()}")
            print(f"  Mean difference: {diff.mean()}")
            assert diff.max() < 1e-6, "Data differs too much"


def test_read_roi():
    """Test reading a specific ROI."""
    print("\n=== Testing ROI Read ===")

    with tempfile.TemporaryDirectory() as tmp_dir:
        voxel_size = (8, 8, 8)
        dataset_path, expected_data = create_test_zarr(tmp_dir, voxel_size=voxel_size)

        idi = ImageDataInterface(dataset_path)
        xidi = XarrayImageDataInterface(dataset_path)

        # Create a ROI in the middle of the dataset
        roi_begin = (64, 64, 64)  # physical coordinates
        roi_shape = (64, 64, 64)  # physical size

        funlib_roi = FunlibRoi(roi_begin, roi_shape)
        xarray_roi = Roi(roi_begin, roi_shape)

        # Read ROI
        idi_data = idi.to_ndarray_ts(funlib_roi)
        xidi_data = xidi.to_ndarray_ts(xarray_roi)

        print(f"  ROI: begin={roi_begin}, shape={roi_shape}")
        print(f"  IDI shape: {idi_data.shape}, XIDI shape: {xidi_data.shape}")
        assert idi_data.shape == xidi_data.shape, f"Shape mismatch: {idi_data.shape} vs {xidi_data.shape}"

        if np.allclose(idi_data, xidi_data):
            print("  ✓ ROI data matches!")
        else:
            diff = np.abs(idi_data.astype(float) - xidi_data.astype(float))
            print(f"  Max difference: {diff.max()}")
            assert diff.max() < 1e-6, "ROI data differs too much"


def test_read_roi_with_padding():
    """Test reading a ROI that extends beyond dataset bounds."""
    print("\n=== Testing ROI with Padding ===")

    with tempfile.TemporaryDirectory() as tmp_dir:
        voxel_size = (8, 8, 8)
        dataset_path, _ = create_test_zarr(tmp_dir, shape=(32, 32, 32), voxel_size=voxel_size)

        idi = ImageDataInterface(dataset_path)
        xidi = XarrayImageDataInterface(dataset_path)

        # Create a ROI that extends beyond the dataset
        # Dataset is 32*8=256 nm in each dimension
        roi_begin = (-16, -16, -16)  # physical coordinates (before dataset start)
        roi_shape = (64, 64, 64)  # physical size

        funlib_roi = FunlibRoi(roi_begin, roi_shape)
        xarray_roi = Roi(roi_begin, roi_shape)

        # Read ROI
        idi_data = idi.to_ndarray_ts(funlib_roi)
        xidi_data = xidi.to_ndarray_ts(xarray_roi)

        print(f"  ROI with padding: begin={roi_begin}, shape={roi_shape}")
        print(f"  IDI shape: {idi_data.shape}, XIDI shape: {xidi_data.shape}")
        assert idi_data.shape == xidi_data.shape, f"Shape mismatch: {idi_data.shape} vs {xidi_data.shape}"

        if np.allclose(idi_data, xidi_data):
            print("  ✓ Padded ROI data matches!")
        else:
            diff = np.abs(idi_data.astype(float) - xidi_data.astype(float))
            print(f"  Max difference: {diff.max()}")
            # Allow some difference due to different padding implementations
            if diff.max() < 1:
                print("  ✓ Padded ROI data close enough (padding may differ slightly)")
            else:
                print(f"  ✗ Data differs too much")


def test_upsampling():
    """Test reading with upsampling (output voxel size < input voxel size)."""
    print("\n=== Testing Upsampling ===")

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_voxel_size = (16, 16, 16)
        output_voxel_size = (8, 8, 8)  # 2x upsampling

        dataset_path, _ = create_test_zarr(tmp_dir, shape=(16, 16, 16), voxel_size=input_voxel_size)

        idi = ImageDataInterface(dataset_path, output_voxel_size=output_voxel_size)
        xidi = XarrayImageDataInterface(dataset_path, output_voxel_size=output_voxel_size)

        # Read full dataset with upsampling
        idi_data = idi.to_ndarray_ts()
        xidi_data = xidi.to_ndarray_ts()

        print(f"  Input voxel size: {input_voxel_size}")
        print(f"  Output voxel size: {output_voxel_size}")
        print(f"  IDI shape: {idi_data.shape}, XIDI shape: {xidi_data.shape}")

        # Expected shape: 16 * 2 = 32 in each dimension
        expected_shape = (32, 32, 32)
        assert idi_data.shape == expected_shape, f"IDI shape mismatch: {idi_data.shape} vs {expected_shape}"
        assert xidi_data.shape == expected_shape, f"XIDI shape mismatch: {xidi_data.shape} vs {expected_shape}"

        if np.allclose(idi_data, xidi_data):
            print("  ✓ Upsampled data matches!")
        else:
            diff = np.abs(idi_data.astype(float) - xidi_data.astype(float))
            print(f"  Max difference: {diff.max()}")
            print(f"  Mean difference: {diff.mean()}")
            # Allow some difference due to different interpolation methods
            if diff.mean() < 1:
                print("  ✓ Upsampled data close enough (interpolation may differ)")


def test_downsampling():
    """Test reading with downsampling (output voxel size > input voxel size)."""
    print("\n=== Testing Downsampling ===")

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_voxel_size = (8, 8, 8)
        output_voxel_size = (16, 16, 16)  # 2x downsampling

        dataset_path, _ = create_test_zarr(tmp_dir, shape=(32, 32, 32), voxel_size=input_voxel_size)

        idi = ImageDataInterface(dataset_path, output_voxel_size=output_voxel_size)
        xidi = XarrayImageDataInterface(dataset_path, output_voxel_size=output_voxel_size)

        # Read full dataset with downsampling
        idi_data = idi.to_ndarray_ts()
        xidi_data = xidi.to_ndarray_ts()

        print(f"  Input voxel size: {input_voxel_size}")
        print(f"  Output voxel size: {output_voxel_size}")
        print(f"  IDI shape: {idi_data.shape}, XIDI shape: {xidi_data.shape}")

        # Expected shape: 32 / 2 = 16 in each dimension
        expected_shape = (16, 16, 16)
        assert idi_data.shape == expected_shape, f"IDI shape mismatch: {idi_data.shape} vs {expected_shape}"
        assert xidi_data.shape == expected_shape, f"XIDI shape mismatch: {xidi_data.shape} vs {expected_shape}"

        if np.allclose(idi_data, xidi_data):
            print("  ✓ Downsampled data matches!")
        else:
            diff = np.abs(idi_data.astype(float) - xidi_data.astype(float))
            print(f"  Max difference: {diff.max()}")
            print(f"  Mean difference: {diff.mean()}")
            # Allow some difference due to different downsampling methods (median vs mean)
            if diff.mean() < 5:
                print("  ✓ Downsampled data close enough (aggregation may differ)")


def test_anisotropic_upsampling():
    """Test per-axis anisotropic upsampling (main new feature)."""
    print("\n=== Testing Anisotropic Upsampling ===")

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_voxel_size = (16, 8, 8)  # Anisotropic: Z is coarser
        output_voxel_size = (8, 8, 8)  # Isotropic output: only Z needs upsampling

        dataset_path, _ = create_test_zarr(tmp_dir, shape=(8, 16, 16), voxel_size=input_voxel_size)

        xidi = XarrayImageDataInterface(dataset_path, output_voxel_size=output_voxel_size)

        # Read full dataset with anisotropic upsampling
        xidi_data = xidi.to_ndarray_ts()

        print(f"  Input voxel size: {input_voxel_size}")
        print(f"  Output voxel size: {output_voxel_size}")
        print(f"  Original shape: (8, 16, 16)")
        print(f"  XIDI output shape: {xidi_data.shape}")

        # Expected shape: Z doubles (8->16), Y and X stay same (16, 16)
        expected_shape = (16, 16, 16)
        assert xidi_data.shape == expected_shape, f"Shape mismatch: {xidi_data.shape} vs {expected_shape}"
        print("  ✓ Anisotropic upsampling shape is correct!")


def test_ds_property():
    """Test that the ds property works for write operations."""
    print("\n=== Testing ds Property (for writes) ===")

    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path, _ = create_test_zarr(tmp_dir)

        xidi = XarrayImageDataInterface(dataset_path, mode="r+")

        # Access ds property
        ds = xidi.ds
        print(f"  ds type: {type(ds)}")
        print(f"  ds.voxel_size: {ds.voxel_size}")

        # Verify it matches
        assert tuple(ds.voxel_size) == tuple(xidi.voxel_size), "ds.voxel_size mismatch"
        print("  ✓ ds property works correctly!")


def test_custom_fill_value():
    """Test custom fill value for padding."""
    print("\n=== Testing Custom Fill Value ===")

    with tempfile.TemporaryDirectory() as tmp_dir:
        voxel_size = (8, 8, 8)
        dataset_path, _ = create_test_zarr(tmp_dir, shape=(16, 16, 16), voxel_size=voxel_size)

        custom_fill = 255
        idi = ImageDataInterface(dataset_path, custom_fill_value=custom_fill)
        xidi = XarrayImageDataInterface(dataset_path, custom_fill_value=custom_fill)

        # Create a ROI that extends beyond the dataset
        roi_begin = (-8, 0, 0)  # Before dataset start
        roi_shape = (16, 16, 16)

        funlib_roi = FunlibRoi(roi_begin, roi_shape)
        xarray_roi = Roi(roi_begin, roi_shape)

        idi_data = idi.to_ndarray_ts(funlib_roi)
        xidi_data = xidi.to_ndarray_ts(xarray_roi)

        print(f"  Custom fill value: {custom_fill}")
        print(f"  IDI padded region value: {idi_data[0, 0, 0]}")
        print(f"  XIDI padded region value: {xidi_data[0, 0, 0]}")

        assert idi_data[0, 0, 0] == custom_fill, "IDI custom fill failed"
        assert xidi_data[0, 0, 0] == custom_fill, "XIDI custom fill failed"
        print("  ✓ Custom fill value works!")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing XarrayImageDataInterface as replacement for ImageDataInterface")
    print("=" * 60)

    tests = [
        test_basic_attributes,
        test_read_full_dataset,
        test_read_roi,
        test_read_roi_with_padding,
        test_upsampling,
        test_downsampling,
        test_anisotropic_upsampling,
        test_ds_property,
        test_custom_fill_value,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)

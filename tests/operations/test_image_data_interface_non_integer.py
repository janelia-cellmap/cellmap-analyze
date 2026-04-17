import json
import os
import struct

import numpy as np
import pytest
from cellmap_analyze.util.image_data_interface import (
    ImageDataInterface,
    is_close_to_integer,
    requires_interpolation,
)
from funlib.geometry import Roi, Coordinate
from cellmap_analyze.util.zarr_io import prepare_ds


class TestHelperFunctions:
    """Test the helper functions for detecting non-integer scale factors."""

    def test_is_close_to_integer_true(self):
        """Test values that are close to integers."""
        assert is_close_to_integer(2.0)
        assert is_close_to_integer(1.999)
        assert is_close_to_integer(2.001)

    def test_is_close_to_integer_false(self):
        """Test values that are not close to integers."""
        assert not is_close_to_integer(0.5)  # 0.5 is not close to an integer
        assert not is_close_to_integer(0.499)
        assert not is_close_to_integer(1.6)
        assert not is_close_to_integer(2.5)
        assert not is_close_to_integer(0.625)
        assert not is_close_to_integer(1.05)

    def test_requires_interpolation_true(self):
        """Test that non-integer factors require interpolation."""
        assert requires_interpolation((1.6, 1.6, 1.6))
        assert requires_interpolation((2.0, 1.6, 2.0))  # Mixed
        assert requires_interpolation((0.625, 0.625, 0.625))
        assert requires_interpolation((0.5, 0.5, 0.5))  # 0.5 is not close to integer

    def test_requires_interpolation_false(self):
        """Test that integer factors don't require interpolation."""
        assert not requires_interpolation((2.0, 2.0, 2.0))
        assert not requires_interpolation((1.999, 2.001, 2.0))  # Close to integer


class TestNonIntegerResampling:
    """Test resampling with non-integer voxel size ratios."""

    def test_non_integer_upsampling(self, tmp_zarr):
        """Test that non-integer upsampling produces correct dimensions."""
        # Create test data at 8nm resolution
        input_voxel_size = Coordinate((8, 8, 8))
        output_voxel_size = Coordinate((5, 5, 5))  # Factor 1.6x upsampling
        test_data = np.arange(1, 28).reshape((3, 3, 3)).astype(np.uint8)

        # Create ROI for dataset
        total_roi = Roi((0, 0, 0), test_data.shape) * input_voxel_size

        # Create dataset
        ds_path = f"{tmp_zarr}/test_upsample/s0"
        ds = prepare_ds(
            tmp_zarr,
            "test_upsample/s0",
            total_roi=total_roi,
            voxel_size=input_voxel_size,
            dtype=test_data.dtype,
        )
        ds[total_roi] = test_data

        # Read with output voxel size
        idi = ImageDataInterface(ds_path, output_voxel_size=output_voxel_size)
        result = idi.to_ndarray_ts()

        # Expected shape: 3 voxels * 8nm / 5nm = 4.8 voxels per axis
        # zoom will produce approximately this size
        expected_shape_approx = tuple(
            int(test_data.shape[i] * input_voxel_size[i] / output_voxel_size[i])
            for i in range(3)
        )

        # Check shape is close to expected (within 1 voxel due to rounding)
        for i in range(3):
            assert abs(result.shape[i] - expected_shape_approx[i]) <= 1

        # Verify labels preserved (no new values introduced)
        assert set(np.unique(result)).issubset(set(np.unique(test_data)))

    def test_non_integer_downsampling(self, tmp_zarr):
        """Test that non-integer downsampling produces correct dimensions."""
        # Create test data at 4nm resolution
        input_voxel_size = Coordinate((4, 4, 4))
        output_voxel_size = Coordinate((10, 10, 10))  # Factor 0.4x downsampling
        test_data = np.random.randint(1, 10, (10, 10, 10), dtype=np.uint8)

        # Create ROI for dataset
        total_roi = Roi((0, 0, 0), test_data.shape) * input_voxel_size

        # Create dataset
        ds_path = f"{tmp_zarr}/test_downsample/s0"
        ds = prepare_ds(
            tmp_zarr,
            "test_downsample/s0",
            total_roi=total_roi,
            voxel_size=input_voxel_size,
            dtype=test_data.dtype,
        )
        ds[total_roi] = test_data

        # Read with output voxel size
        idi = ImageDataInterface(ds_path, output_voxel_size=output_voxel_size)
        result = idi.to_ndarray_ts()

        # Expected shape: 10 voxels * 4nm / 10nm = 4 voxels per axis
        expected_shape = tuple(
            int(test_data.shape[i] * input_voxel_size[i] / output_voxel_size[i])
            for i in range(3)
        )

        # Check shape matches expected
        for i in range(3):
            assert abs(result.shape[i] - expected_shape[i]) <= 1

        # Verify labels preserved
        assert set(np.unique(result)).issubset(set(np.unique(test_data)))

    def test_anisotropic_non_integer(self, tmp_zarr):
        """Test mixed integer and non-integer factors per axis."""
        input_voxel_size = Coordinate((8, 10, 5))
        output_voxel_size = Coordinate((5, 4, 5))  # Factors: (1.6, 2.5, 1.0)
        test_data = np.random.randint(1, 10, (4, 4, 4), dtype=np.uint8)

        # Create ROI for dataset
        total_roi = Roi((0, 0, 0), test_data.shape) * input_voxel_size

        # Create dataset
        ds_path = f"{tmp_zarr}/test_anisotropic/s0"
        ds = prepare_ds(
            tmp_zarr,
            "test_anisotropic/s0",
            total_roi=total_roi,
            voxel_size=input_voxel_size,
            dtype=test_data.dtype,
        )
        ds[total_roi] = test_data

        # Read with output voxel size
        idi = ImageDataInterface(ds_path, output_voxel_size=output_voxel_size)
        result = idi.to_ndarray_ts()

        # Expected shape calculation per axis
        expected_shape = tuple(
            int(test_data.shape[i] * input_voxel_size[i] / output_voxel_size[i])
            for i in range(3)
        )

        # Check shape is close to expected
        for i in range(3):
            assert abs(result.shape[i] - expected_shape[i]) <= 1

    def test_label_preservation_order_0(self, tmp_zarr):
        """Verify no new label values introduced with order=0."""
        input_voxel_size = Coordinate((8, 8, 8))
        output_voxel_size = Coordinate((5, 5, 5))
        # Create data with specific labels
        test_data = np.array([[[1, 2, 1],
                               [2, 3, 2],
                               [1, 2, 1]],
                              [[2, 3, 2],
                               [3, 4, 3],
                               [2, 3, 2]],
                              [[1, 2, 1],
                               [2, 3, 2],
                               [1, 2, 1]]], dtype=np.uint8)

        # Create ROI for dataset
        total_roi = Roi((0, 0, 0), test_data.shape) * input_voxel_size

        # Create dataset
        ds_path = f"{tmp_zarr}/test_labels/s0"
        ds = prepare_ds(
            tmp_zarr,
            "test_labels/s0",
            total_roi=total_roi,
            voxel_size=input_voxel_size,
            dtype=test_data.dtype,
        )
        ds[total_roi] = test_data

        # Read with interpolation_order=0 (nearest-neighbor)
        idi = ImageDataInterface(
            ds_path,
            output_voxel_size=output_voxel_size,
            interpolation_order=0
        )
        result = idi.to_ndarray_ts()

        # Original labels: {1, 2, 3, 4}
        original_labels = set(np.unique(test_data))
        result_labels = set(np.unique(result))

        # Result should only contain original labels (no fuzzy interpolation)
        assert result_labels.issubset(original_labels)

    def test_roi_extraction_non_integer(self, tmp_zarr):
        """Test that ROI extraction works correctly with non-integer scaling."""
        input_voxel_size = Coordinate((8, 8, 8))
        output_voxel_size = Coordinate((5, 5, 5))
        test_data = np.random.randint(1, 10, (10, 10, 10), dtype=np.uint8)

        # Create ROI for dataset
        total_roi = Roi((0, 0, 0), test_data.shape) * input_voxel_size

        # Create dataset
        ds_path = f"{tmp_zarr}/test_roi_non_integer/s0"
        ds = prepare_ds(
            tmp_zarr,
            "test_roi_non_integer/s0",
            total_roi=total_roi,
            voxel_size=input_voxel_size,
            dtype=test_data.dtype,
        )
        ds[total_roi] = test_data

        # Read a specific ROI in physical coordinates
        roi = Roi((16, 16, 16), (32, 32, 32))  # Physical coordinates (nm)
        idi = ImageDataInterface(ds_path, output_voxel_size=output_voxel_size)
        result = idi.to_ndarray_ts(roi)

        # Expected shape: 32nm / 5nm = 6.4 voxels per axis
        expected_shape = tuple(int(s / vs) for s, vs in zip(roi.shape, output_voxel_size))

        # Check shape is close
        for i in range(3):
            assert abs(result.shape[i] - expected_shape[i]) <= 1


class TestIntegerFastPaths:
    """Test that integer scale factors still use fast paths."""

    def test_integer_upsampling_uses_repeat(self, tmp_zarr):
        """Test that integer factors use repeat() not zoom()."""
        input_voxel_size = Coordinate((8, 8, 8))
        output_voxel_size = Coordinate((4, 4, 4))  # Exact 2x upsampling
        test_data = np.arange(1, 28).reshape((3, 3, 3)).astype(np.uint8)

        # Create ROI for dataset
        total_roi = Roi((0, 0, 0), test_data.shape) * input_voxel_size

        ds_path = f"{tmp_zarr}/test_int_upsample/s0"
        ds = prepare_ds(
            tmp_zarr,
            "test_int_upsample/s0",
            total_roi=total_roi,
            voxel_size=input_voxel_size,
            dtype=test_data.dtype,
        )
        ds[total_roi] = test_data

        idi = ImageDataInterface(ds_path, output_voxel_size=output_voxel_size)
        result = idi.to_ndarray_ts()

        # Should produce exactly 2x the original size
        expected_shape = tuple(s * 2 for s in test_data.shape)
        assert result.shape == expected_shape

    def test_integer_downsampling_uses_slicing(self, tmp_zarr):
        """Test that integer downsampling uses slicing."""
        input_voxel_size = Coordinate((4, 4, 4))
        output_voxel_size = Coordinate((8, 8, 8))  # Exact 2x downsampling
        test_data = np.random.randint(1, 10, (10, 10, 10), dtype=np.uint8)

        # Create ROI for dataset
        total_roi = Roi((0, 0, 0), test_data.shape) * input_voxel_size

        ds_path = f"{tmp_zarr}/test_int_downsample/s0"
        ds = prepare_ds(
            tmp_zarr,
            "test_int_downsample/s0",
            total_roi=total_roi,
            voxel_size=input_voxel_size,
            dtype=test_data.dtype,
        )
        ds[total_roi] = test_data

        idi = ImageDataInterface(ds_path, output_voxel_size=output_voxel_size)
        result = idi.to_ndarray_ts()

        # Should produce exactly half the original size
        expected_shape = tuple(s // 2 for s in test_data.shape)
        assert result.shape == expected_shape

        # Labels should be preserved exactly (slicing doesn't aggregate)
        assert set(np.unique(result)).issubset(set(np.unique(test_data)))

    def test_close_to_integer_uses_fast_path(self, tmp_zarr):
        """Test that factors close to integers (1.999) use fast paths."""
        input_voxel_size = Coordinate((8.004, 8.004, 8.004))  # Close to 8
        output_voxel_size = Coordinate((4, 4, 4))  # Factor ~2.001
        test_data = np.arange(1, 28).reshape((3, 3, 3)).astype(np.uint8)

        # Create ROI for dataset
        total_roi = Roi((0, 0, 0), test_data.shape) * input_voxel_size

        ds_path = f"{tmp_zarr}/test_close_int/s0"
        ds = prepare_ds(
            tmp_zarr,
            "test_close_int/s0",
            total_roi=total_roi,
            voxel_size=input_voxel_size,
            dtype=test_data.dtype,
        )
        ds[total_roi] = test_data

        idi = ImageDataInterface(ds_path, output_voxel_size=output_voxel_size)
        result = idi.to_ndarray_ts()

        # Should use repeat and produce 2x size
        expected_shape = tuple(s * 2 for s in test_data.shape)
        assert result.shape == expected_shape


class TestVoxelCoverageAccuracy:
    """Test that physical voxel coverage is correct after resampling."""

    def test_physical_dimensions_preserved(self, tmp_zarr):
        """Verify that physical dimensions are preserved through resampling."""
        input_voxel_size = Coordinate((8, 8, 8))
        output_voxel_size = Coordinate((5, 5, 5))
        test_shape = (10, 10, 10)
        test_data = np.random.randint(1, 10, test_shape, dtype=np.uint8)

        # Create ROI for dataset
        total_roi = Roi((0, 0, 0), test_shape) * input_voxel_size

        ds_path = f"{tmp_zarr}/test_physical_dims/s0"
        ds = prepare_ds(
            tmp_zarr,
            "test_physical_dims/s0",
            total_roi=total_roi,
            voxel_size=input_voxel_size,
            dtype=test_data.dtype,
        )
        ds[total_roi] = test_data

        idi = ImageDataInterface(ds_path, output_voxel_size=output_voxel_size)
        result = idi.to_ndarray_ts()

        # Physical size in nm
        input_physical_size = tuple(s * vs for s, vs in zip(test_shape, input_voxel_size))
        output_physical_size = tuple(s * vs for s, vs in zip(result.shape, output_voxel_size))

        # Physical sizes should match (within 1 voxel tolerance)
        for i in range(3):
            # Allow up to 1 output voxel difference
            assert abs(input_physical_size[i] - output_physical_size[i]) <= output_voxel_size[i]


class TestCrossDatasetAlignment:
    """Test exact shape matching across datasets with different input voxel sizes."""

    def test_exact_shape_match_different_voxel_sizes(self, tmp_zarr):
        """Verify datasets with different input voxel sizes return identical shapes.

        This is the critical test for global grid alignment. When two datasets with
        different input voxel sizes are resampled to the same output voxel size and
        queried with the same ROI, they MUST return exactly the same shape.
        """
        # Use the specific example from the user's question
        # ROI: (137, 1449, 1424) + (173, 142, 47) nm
        # Dataset A: voxel size (7, 11, 13)
        # Dataset B: voxel size (5, 9, 17)
        # Output: (3, 3, 3)

        output_voxel_size = Coordinate((3, 3, 3))

        # ROI aligned to output voxel grid (multiples of 3)
        # Use simpler coordinates that fit within reasonable dataset sizes
        # Start: (138, 144, 150) - all multiples of 3
        # Size: (171, 141, 45) - all multiples of 3
        roi_begin = Coordinate((138, 144, 150))
        roi_size = Coordinate((171, 141, 45))
        roi = Roi(roi_begin, roi_size)

        # Dataset A with voxel size (7, 11, 13)
        input_voxel_size_A = Coordinate((7, 11, 13))
        # Need to cover ROI (138, 144, 150) + (171, 141, 45) = up to (309, 285, 195)
        # In voxels: ceil(309/7)=45, ceil(285/11)=26, ceil(195/13)=15
        test_shape_A = (45, 26, 15)
        test_data_A = np.random.randint(1, 10, test_shape_A, dtype=np.uint8)
        total_roi_A = Roi((0, 0, 0), test_shape_A) * input_voxel_size_A

        ds_path_A = f"{tmp_zarr}/test_dataset_A/s0"
        ds_A = prepare_ds(
            tmp_zarr,
            "test_dataset_A/s0",
            total_roi=total_roi_A,
            voxel_size=input_voxel_size_A,
            dtype=test_data_A.dtype,
        )
        ds_A[total_roi_A] = test_data_A

        # Dataset B with voxel size (5, 9, 17)
        input_voxel_size_B = Coordinate((5, 9, 17))
        # Need to cover same ROI: ceil(309/5)=62, ceil(285/9)=32, ceil(195/17)=12
        test_shape_B = (62, 32, 12)
        test_data_B = np.random.randint(1, 10, test_shape_B, dtype=np.uint8)
        total_roi_B = Roi((0, 0, 0), test_shape_B) * input_voxel_size_B

        ds_path_B = f"{tmp_zarr}/test_dataset_B/s0"
        ds_B = prepare_ds(
            tmp_zarr,
            "test_dataset_B/s0",
            total_roi=total_roi_B,
            voxel_size=input_voxel_size_B,
            dtype=test_data_B.dtype,
        )
        ds_B[total_roi_B] = test_data_B

        # Query both datasets with the same ROI at the same output voxel size
        idi_A = ImageDataInterface(ds_path_A, output_voxel_size=output_voxel_size)
        result_A = idi_A.to_ndarray_ts(roi)

        idi_B = ImageDataInterface(ds_path_B, output_voxel_size=output_voxel_size)
        result_B = idi_B.to_ndarray_ts(roi)

        # CRITICAL: Shapes must be EXACTLY the same (not ±1 voxel)
        print(f"Dataset A shape: {result_A.shape}")
        print(f"Dataset B shape: {result_B.shape}")
        assert result_A.shape == result_B.shape, (
            f"Shapes must be identical for exact alignment! "
            f"Got {result_A.shape} vs {result_B.shape}"
        )

        # Expected shape should be roi_size / output_voxel_size
        expected_shape = tuple(
            int(roi_size[i] / output_voxel_size[i]) for i in range(3)
        )
        assert result_A.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {result_A.shape}"
        )


def _create_n5_dataset(base_path, shape, block_size, voxel_size_xyz, data):
    """Create a minimal N5 dataset with raw chunks that tensorstore can read."""
    os.makedirs(base_path, exist_ok=True)

    attrs = {
        "dimensions": list(shape),
        "blockSize": list(block_size),
        "dataType": "uint8",
        "compression": {"type": "raw"},
        "pixelResolution": {"dimensions": voxel_size_xyz, "unit": "nm"},
    }
    with open(os.path.join(base_path, "attributes.json"), "w") as f:
        json.dump(attrs, f)

    bs = block_size
    for xi in range(0, shape[0], bs[0]):
        for yi in range(0, shape[1], bs[1]):
            for zi in range(0, shape[2], bs[2]):
                chunk_data = data[xi : xi + bs[0], yi : yi + bs[1], zi : zi + bs[2]]
                chunk_path = os.path.join(
                    base_path,
                    str(xi // bs[0]),
                    str(yi // bs[1]),
                    str(zi // bs[2]),
                )
                os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
                # N5 raw chunk: mode(2B) + ndim(2B) + dim_sizes(4B each) + data
                header = struct.pack(">HH", 0, 3)
                for d in chunk_data.shape:
                    header += struct.pack(">I", d)
                with open(chunk_path, "wb") as f:
                    f.write(header)
                    f.write(chunk_data.tobytes())


@pytest.fixture
def tmp_n5(tmp_path):
    """Create an N5 dataset with anisotropic non-integer voxel sizes (4, 4, 5.24)."""
    n5_path = str(tmp_path / "test.n5")
    os.makedirs(n5_path)
    with open(os.path.join(n5_path, "attributes.json"), "w") as f:
        json.dump({"n5": "2.0.0"}, f)

    labels_path = os.path.join(n5_path, "labels")
    os.makedirs(labels_path)
    with open(os.path.join(labels_path, "attributes.json"), "w") as f:
        json.dump({}, f)

    # N5 dimensions/pixelResolution are in x,y,z order
    data = np.random.randint(1, 10, (20, 25, 10), dtype=np.uint8)
    _create_n5_dataset(
        os.path.join(n5_path, "labels", "s0"),
        shape=[20, 25, 10],
        block_size=[10, 10, 10],
        voxel_size_xyz=[4.0, 4.0, 5.24],
        data=data,
    )
    return os.path.join(n5_path, "labels/s0")


class TestN5AnisotropicSwapAxes:
    """Test that N5 datasets with anisotropic voxel sizes read correct shapes.

    N5 format requires axis swapping (swap_axes=True). When voxel sizes are
    anisotropic, the swap must also apply to voxel_size so that snap_to_grid
    uses the correct component per axis.
    """

    def test_single_voxel_roi_z_axis(self, tmp_n5):
        """A 1-voxel-thick ROI in z should return shape 1 in that dimension."""
        idi = ImageDataInterface(tmp_n5)
        # IDI voxel_size is now (131, 100, 100) in z,y,x; z-voxel = 131
        roi = Roi(Coordinate(0, 0, 0), Coordinate(131, 1000, 1000))
        result = idi.to_ndarray_ts(roi)
        assert result.shape[0] == 1, (
            f"Expected 1 voxel in z, got {result.shape[0]} (shape={result.shape})"
        )

    def test_single_voxel_roi_x_axis(self, tmp_n5):
        """A 1-voxel-thick ROI in x should return shape 1 in that dimension."""
        idi = ImageDataInterface(tmp_n5)
        # x-voxel = 100
        roi = Roi(Coordinate(0, 0, 0), Coordinate(1310, 1000, 100))
        result = idi.to_ndarray_ts(roi)
        assert result.shape[2] == 1, (
            f"Expected 1 voxel in x, got {result.shape[2]} (shape={result.shape})"
        )

    def test_roi_beyond_dataset_returns_correct_axes(self, tmp_n5):
        """A ROI fully outside the dataset should return zeros in z,y,x order."""
        idi = ImageDataInterface(tmp_n5)
        # ROI completely past the dataset end in z; should be 1 voxel in z
        roi = Roi(
            Coordinate(idi.roi.end[0], 0, 0),
            Coordinate(131, 1000, 1000),
        )
        result = idi.to_ndarray_ts(roi)
        assert result.shape[0] == 1, (
            f"Expected 1 voxel in z (no-overlap path), got shape {result.shape}"
        )
        assert np.all(result == 0)

    def test_full_dataset_no_roi(self, tmp_n5):
        """Reading full dataset (roi=None) should return correct axis order."""
        idi = ImageDataInterface(tmp_n5)
        result = idi.to_ndarray_ts()
        # N5 on-disk shape is (20,25,10) in x,y,z; IDI presents as (10,25,20) in z,y,x
        assert result.shape == (10, 25, 20)

    def test_full_roi_shape(self, tmp_n5):
        """Full dataset ROI should return the correct array shape."""
        idi = ImageDataInterface(tmp_n5)
        result = idi.to_ndarray_ts(idi.roi)
        # N5 on-disk shape is (20,25,10) in x,y,z; IDI presents as (10,25,20) in z,y,x
        assert result.shape == (10, 25, 20)

import json
import os
import struct

import numpy as np
import pytest
import worldarray as wa
from cellmap_analyze.util.grid_util import make_grid
from cellmap_analyze.util.zarr_util import create_multiscale_dataset, open_dataset


def _write(path, data, voxel_size):
    """``data`` as s0 of an OME-Zarr group at ``path``, in one chunk, voxel 0's corner
    at the origin."""
    vol = create_multiscale_dataset(
        path, data.dtype, make_grid(data.shape, voxel_size, chunks=data.shape)
    )
    vol.write_index((0, 0, 0), vol.shape, data)


def _read_at(path, voxel_size, **kwargs):
    """The dataset at ``path`` read onto ``voxel_size`` voxels covering it."""
    vol = open_dataset(path, **kwargs)
    return vol.read_onto(wa.Grid.from_meta(vol.meta, voxel_size=voxel_size))


class TestNonIntegerResampling:
    """Test resampling with non-integer voxel size ratios."""

    def test_non_integer_upsampling(self, tmp_zarr):
        """Test that non-integer upsampling produces correct dimensions."""
        input_voxel_size = (8, 8, 8)
        output_voxel_size = (5, 5, 5)  # Factor 1.6x upsampling
        test_data = np.arange(1, 28).reshape((3, 3, 3)).astype(np.uint8)
        _write(f"{tmp_zarr}/test_upsample", test_data, input_voxel_size)

        result = _read_at(f"{tmp_zarr}/test_upsample/s0", output_voxel_size)

        # 3 voxels * 8nm / 5nm = 4.8 voxels per axis, so 5 cover it
        assert result.shape == (5, 5, 5)
        # Verify labels preserved (no new values introduced)
        assert set(np.unique(result)).issubset(set(np.unique(test_data)))

    def test_non_integer_downsampling(self, tmp_zarr):
        """Test that non-integer downsampling produces correct dimensions."""
        input_voxel_size = (4, 4, 4)
        output_voxel_size = (10, 10, 10)  # Factor 0.4x downsampling
        test_data = np.random.randint(1, 10, (10, 10, 10), dtype=np.uint8)
        _write(f"{tmp_zarr}/test_downsample", test_data, input_voxel_size)

        result = _read_at(f"{tmp_zarr}/test_downsample/s0", output_voxel_size)

        # 10 voxels * 4nm / 10nm = 4 voxels per axis
        assert result.shape == (4, 4, 4)
        assert set(np.unique(result)).issubset(set(np.unique(test_data)))

    def test_anisotropic_non_integer(self, tmp_zarr):
        """Test mixed integer and non-integer factors per axis."""
        input_voxel_size = (8, 10, 5)
        output_voxel_size = (5, 4, 5)  # Factors: (1.6, 2.5, 1.0)
        test_data = np.random.randint(1, 10, (4, 4, 4), dtype=np.uint8)
        _write(f"{tmp_zarr}/test_anisotropic", test_data, input_voxel_size)

        result = _read_at(f"{tmp_zarr}/test_anisotropic/s0", output_voxel_size)

        # ceil(32/5), 40/4, 20/5
        assert result.shape == (7, 10, 4)

    def test_label_preservation_order_0(self, tmp_zarr):
        """Verify no new label values introduced with order=0."""
        input_voxel_size = (8, 8, 8)
        output_voxel_size = (5, 5, 5)
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
        _write(f"{tmp_zarr}/test_labels", test_data, input_voxel_size)

        # Read nearest-neighbor
        result = _read_at(
            f"{tmp_zarr}/test_labels/s0", output_voxel_size, method="nearest"
        )

        # Result should only contain original labels (no fuzzy interpolation)
        assert set(np.unique(result)).issubset(set(np.unique(test_data)))

    def test_roi_extraction_non_integer(self, tmp_zarr):
        """A box of output voxels from a nm corner reads that many voxels."""
        test_data = np.random.randint(1, 10, (10, 10, 10), dtype=np.uint8)
        _write(f"{tmp_zarr}/test_roi_non_integer", test_data, (8, 8, 8))

        vol = open_dataset(f"{tmp_zarr}/test_roi_non_integer/s0")
        # 6 voxels of 5 nm from 16 nm, which does not sit on the 8 nm grid
        result = vol.read_onto(make_grid((6, 6, 6), (5, 5, 5), corner=(16, 16, 16)))
        assert result.shape == (6, 6, 6)
        # nearest: output voxel i is centred at 16 + 5 i + 2.5 nm
        source = ((16 + 5 * np.arange(6) + 2.5) // 8).astype(int)
        assert np.array_equal(result, test_data[np.ix_(source, source, source)])


class TestIntegerRatios:
    """Integer voxel-size ratios repeat or pick voxels exactly."""

    def test_integer_upsampling_repeats(self, tmp_zarr):
        test_data = np.arange(1, 28).reshape((3, 3, 3)).astype(np.uint8)
        _write(f"{tmp_zarr}/test_int_upsample", test_data, (8, 8, 8))

        result = _read_at(f"{tmp_zarr}/test_int_upsample/s0", (4, 4, 4))

        assert np.array_equal(result, test_data.repeat(2, 0).repeat(2, 1).repeat(2, 2))

    def test_integer_downsampling_picks_voxels(self, tmp_zarr):
        test_data = np.random.randint(1, 10, (10, 10, 10), dtype=np.uint8)
        _write(f"{tmp_zarr}/test_int_downsample", test_data, (4, 4, 4))

        result = _read_at(f"{tmp_zarr}/test_int_downsample/s0", (8, 8, 8))

        assert result.shape == (5, 5, 5)
        # Labels should be preserved exactly (nearest doesn't aggregate)
        assert set(np.unique(result)).issubset(set(np.unique(test_data)))

    def test_close_to_integer_ratio_repeats(self, tmp_zarr):
        """A ratio of ~2.001 still gives every voxel twice over the 2x grid."""
        test_data = np.arange(1, 28).reshape((3, 3, 3)).astype(np.uint8)
        _write(f"{tmp_zarr}/test_close_int", test_data, (8.004, 8.004, 8.004))

        vol = open_dataset(f"{tmp_zarr}/test_close_int/s0")
        result = vol.read_onto(make_grid((6, 6, 6), (4, 4, 4)))

        assert np.array_equal(result, test_data.repeat(2, 0).repeat(2, 1).repeat(2, 2))


class TestVoxelCoverageAccuracy:
    """Test that physical voxel coverage is correct after resampling."""

    def test_physical_dimensions_preserved(self, tmp_zarr):
        """Verify that physical dimensions are preserved through resampling."""
        input_voxel_size = (8, 8, 8)
        output_voxel_size = (5, 5, 5)
        test_shape = (10, 10, 10)
        test_data = np.random.randint(1, 10, test_shape, dtype=np.uint8)
        _write(f"{tmp_zarr}/test_physical_dims", test_data, input_voxel_size)

        result = _read_at(f"{tmp_zarr}/test_physical_dims/s0", output_voxel_size)

        # Physical sizes should match (within 1 output voxel)
        for i in range(3):
            input_size = test_shape[i] * input_voxel_size[i]
            output_size = result.shape[i] * output_voxel_size[i]
            assert 0 <= output_size - input_size < output_voxel_size[i]


class TestCrossDatasetAlignment:
    """Datasets with different voxel sizes read onto one grid line up voxel for voxel."""

    def test_exact_shape_match_different_voxel_sizes(self, tmp_zarr):
        # 3 nm voxels over (138, 144, 150) + (171, 141, 45) nm
        grid = make_grid((57, 47, 15), (3, 3, 3), corner=(138, 144, 150))

        # Dataset A with voxel size (7, 11, 13), covering the grid
        test_data_A = np.random.randint(1, 10, (45, 26, 15), dtype=np.uint8)
        _write(f"{tmp_zarr}/test_dataset_A", test_data_A, (7, 11, 13))
        # Dataset B with voxel size (5, 9, 17)
        test_data_B = np.random.randint(1, 10, (62, 32, 12), dtype=np.uint8)
        _write(f"{tmp_zarr}/test_dataset_B", test_data_B, (5, 9, 17))

        results = []
        for name, data, vs in (("A", test_data_A, (7, 11, 13)), ("B", test_data_B, (5, 9, 17))):
            result = open_dataset(f"{tmp_zarr}/test_dataset_{name}/s0").read_onto(grid)
            # nearest: each 3 nm voxel takes the source voxel its centre is in
            centres = [138 + 3 * np.arange(57) + 1.5, 144 + 3 * np.arange(47) + 1.5, 150 + 3 * np.arange(15) + 1.5]
            source = [(c // v).astype(int) for c, v in zip(centres, vs)]
            assert np.array_equal(result, data[np.ix_(*source)])
            results.append(result)
        assert results[0].shape == results[1].shape == grid.shape


class TestOffsetAlignment:
    """Reads land on the output grid's voxels wherever the dataset's corner is."""

    def test_exact_single_native_voxel_box_returns_that_voxel(self, tmp_zarr):
        test_data = np.arange(10, dtype=np.uint8).reshape(10, 1, 1)
        _write(f"{tmp_zarr}/test_offset_align", test_data, (32, 1, 1))

        vol = open_dataset(f"{tmp_zarr}/test_offset_align/s0")
        # exactly the physical extent of native voxel 3, in 8 nm voxels: every
        # returned voxel should be 3, not a mix of 2 and 3.
        corner = (vol.corner[0] + 3 * 32, vol.corner[1], vol.corner[2])
        result = vol.read_onto(make_grid((4, 1, 1), (8, 1, 1), corner))
        assert np.array_equal(result.ravel(), np.full(4, 3))


class TestRawValidMask:
    """Regression coverage for return_valid_mask: voxels that come from
    padding/no-overlap fill must be reported as invalid rather than left
    indistinguishable from real (possibly legitimately-zero) data."""

    def _make_dataset(self, tmp_zarr, name, input_voxel_size, shape, fill_start=1):
        test_data = (
            np.arange(fill_start, fill_start + np.prod(shape))
            .reshape(shape)
            .astype(np.uint16)
        )
        _write(f"{tmp_zarr}/{name}", test_data, input_voxel_size)
        return open_dataset(f"{tmp_zarr}/{name}/s0")

    def test_no_overlap_box_is_entirely_invalid(self, tmp_zarr):
        vol = self._make_dataset(tmp_zarr, "test_valid_no_overlap", (32, 32, 32), (5, 5, 5))
        # Entirely past the dataset's own extent -> no real data anywhere.
        grid = make_grid((10, 10, 10), (8, 8, 8), corner=vol.box.end)
        data, valid = vol.read_onto(grid, mask=True)
        assert data.shape == valid.shape
        assert not valid.any()

    def test_fully_inside_box_is_entirely_valid(self, tmp_zarr):
        vol = self._make_dataset(tmp_zarr, "test_valid_inside", (32, 32, 32), (5, 5, 5))
        data, valid = vol.read_onto(wa.Grid.from_meta(vol.meta, voxel_size=(8, 8, 8)), mask=True)
        assert data.shape == valid.shape
        assert valid.all()

    def test_partial_overlap_marks_only_the_covered_region_valid(self, tmp_zarr):
        vol = self._make_dataset(tmp_zarr, "test_valid_partial", (32, 32, 32), (5, 5, 5))
        grid = wa.Grid.from_meta(vol.meta, voxel_size=(8, 8, 8))
        # Straddle the dataset's far edge on axis 0 only: half real data,
        # half padding.
        data, valid = vol.read_onto(grid, (16, 0, 0), (24, 4, 4), mask=True)

        assert data.shape == valid.shape
        assert valid[:4].all()
        assert not valid[4:].any()


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
    """N5 stores x,y,z; the IDI presents z,y,x, voxel sizes included."""

    def test_voxel_size_and_shape_are_zyx(self, tmp_n5):
        vol = open_dataset(tmp_n5)
        assert vol.voxel_size == (5.24, 4.0, 4.0)
        # N5 on-disk shape is (20,25,10) in x,y,z; IDI presents as (10,25,20) in z,y,x
        assert vol.shape == (10, 25, 20)
        assert vol.read_index((0, 0, 0), vol.shape).shape == (10, 25, 20)

    def test_box_beyond_dataset_returns_zeros(self, tmp_n5):
        vol = open_dataset(tmp_n5)
        result = vol.read_index((10, 0, 0), (11, 10, 10))
        assert result.shape == (1, 10, 10)
        assert np.all(result == 0)

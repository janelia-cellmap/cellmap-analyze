"""Integration tests for non-integer voxel size support."""
import numpy as np
import pytest
import worldarray as wa
import zarr

from cellmap_analyze.util.grid_util import grid_of, make_grid
from cellmap_analyze.util.zarr_util import create_multiscale_dataset, open_dataset
from cellmap_analyze.util.measure_util import get_region_properties


def _create(path, shape, voxel_size, dtype=np.uint8):
    return create_multiscale_dataset(path, dtype, make_grid(shape, voxel_size, chunks=shape))


@pytest.fixture
def non_int_zarr(tmp_path):
    """Create a zarr dataset with non-integer voxel_size in attributes."""
    zarr_path = str(tmp_path / "test.zarr")
    original_vs = (3.54, 4.0, 4.0)
    data_shape = (10, 10, 10)
    ds = _create(f"{zarr_path}/test_ds", data_shape, original_vs)

    # Write test data: a simple labeled volume
    test_data = np.zeros(data_shape, dtype=np.uint8)
    test_data[2:5, 2:5, 2:5] = 1  # 27-voxel cube labeled 1
    test_data[6:8, 6:8, 6:8] = 2  # 8-voxel cube labeled 2
    ds.write_index((0, 0, 0), ds.shape, test_data)

    return zarr_path, original_vs, test_data


class TestIDINonIntegerVoxelSize:
    def test_reads_original_voxel_size(self, non_int_zarr):
        zarr_path, original_vs, test_data = non_int_zarr
        vol = open_dataset(f"{zarr_path}/test_ds/s0")

        assert vol.voxel_size == original_vs
        assert grid_of(vol).shape == test_data.shape

    def test_data_read_correct(self, non_int_zarr):
        """Data should be readable without corruption."""
        zarr_path, _, test_data = non_int_zarr
        vol = open_dataset(f"{zarr_path}/test_ds/s0")

        data = vol.read_onto(grid_of(vol))
        np.testing.assert_array_equal(data, test_data)


class TestMeasurementsWithNonIntegerVoxelSize:
    def test_volume_correct(self, non_int_zarr):
        """Volume should use the true voxel_size."""
        zarr_path, original_vs, test_data = non_int_zarr
        vol = open_dataset(f"{zarr_path}/test_ds/s0")

        df = get_region_properties(test_data, voxel_size=vol.voxel_size, trim=0)

        true_voxel_volume = original_vs[0] * original_vs[1] * original_vs[2]
        # Label 1 has 27 voxels (3x3x3 cube)
        label_1_row = df[df["ID"] == 1].iloc[0]
        assert abs(label_1_row["Volume (nm^3)"] - 27 * true_voxel_volume) < 1e-6

        # Label 2 has 8 voxels (2x2x2 cube)
        label_2_row = df[df["ID"] == 2].iloc[0]
        assert abs(label_2_row["Volume (nm^3)"] - 8 * true_voxel_volume) < 1e-6

    def test_surface_area_correct(self, non_int_zarr):
        """Surface area should use the true voxel_size's face areas."""
        zarr_path, original_vs, test_data = non_int_zarr
        vol = open_dataset(f"{zarr_path}/test_ds/s0")

        df = get_region_properties(test_data, voxel_size=vol.voxel_size, trim=0)

        # For label 2 (2x2x2 cube): 6 faces, each face has 4 exposed voxel faces
        # Face areas: YX = 4*4=16, ZX = 3.54*4=14.16, ZY = 3.54*4=14.16
        yz_area = original_vs[1] * original_vs[2]  # 16
        zx_area = original_vs[0] * original_vs[2]  # 14.16
        zy_area = original_vs[0] * original_vs[1]  # 14.16
        # 2x2x2 cube has 4 exposed faces per side:
        # 2 Z-faces with 4 voxel faces each = 8 * yz_area
        # 2 Y-faces with 4 voxel faces each = 8 * zx_area
        # 2 X-faces with 4 voxel faces each = 8 * zy_area
        expected_sa = 8 * yz_area + 8 * zx_area + 8 * zy_area

        label_2_row = df[df["ID"] == 2].iloc[0]
        assert abs(label_2_row["Surface Area (nm^2)"] - expected_sa) < 1e-3


class TestOutputMetadata:
    def test_ome_zarr_has_original_voxel_size(self, tmp_path):
        """OME-Zarr metadata should have the original float voxel_size."""
        output_path = str(tmp_path / "output.zarr" / "test_ds")
        original_vs = (3.54, 4.0, 4.0)
        _create(output_path, (10, 10, 10), original_vs)

        # Read back the OME-Zarr metadata via zarr API (works for v2 and v3)
        group = zarr.open_group(
            str(tmp_path / "output.zarr" / "test_ds"), mode="r"
        )
        metadata = dict(group.attrs)

        scale_transform = metadata["ome"]["multiscales"][0]["datasets"][0][
            "coordinateTransformations"
        ][0]
        assert scale_transform["type"] == "scale"
        for written, expected in zip(scale_transform["scale"], original_vs):
            assert abs(written - expected) < 1e-10

    def test_no_per_array_placement_attrs(self, tmp_path):
        """Outputs are placed by the group's OME metadata alone: the arrays carry
        no funlib voxel_size/offset attributes."""
        output_path = str(tmp_path / "output2.zarr" / "test_ds")
        _create(output_path, (5, 5, 5), (3.54, 4.0, 4.0))

        assert dict(zarr.open_array(output_path + "/s0", mode="r").attrs) == {}

    def test_written_voxel_size_is_true_physical_value(self, tmp_path):
        """The persisted voxel size is the true physical voxel size."""
        output_path = str(tmp_path / "true_vs.zarr" / "test_ds")
        original_vs = (72.96, 64.0, 64.0)
        _create(output_path, (5, 5, 5), original_vs)

        written = wa.open(output_path + "/s0").voxel_size
        for s, e in zip(written, original_vs):
            assert abs(float(s) - e) < 1e-10, "voxel size should be physical"
        assert open_dataset(output_path + "/s0").voxel_size == original_vs

    @pytest.mark.parametrize(
        "original_vs, translation",
        [((5.0, 5.0, 5.0), (0.0, 0.0, 0.0)), ((8.0, 8.0, 8.0), (4.5, 0.0, 0.0))],
    )
    def test_output_keeps_input_placement(self, tmp_path, original_vs, translation):
        """An output written on its input's grid sits where the input does (here
        with voxel 0's corner at -2.5 nm and 0.5 nm)."""
        group = zarr.open_group(str(tmp_path / "in.zarr"), mode="w")
        group.create_array("s0", shape=(16, 16, 16), chunks=(8, 8, 8), dtype="uint8")
        group.attrs["multiscales"] = [{
            "version": "0.4",
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
            "datasets": [{"path": "s0", "coordinateTransformations": [
                {"type": "scale", "scale": list(original_vs)},
                {"type": "translation", "translation": list(translation)},
            ]}],
        }]
        input_vol = open_dataset(str(tmp_path / "in.zarr" / "s0"))

        output_vol = create_multiscale_dataset(
            str(tmp_path / "out.zarr" / "seg"), dtype=np.uint8, grid=grid_of(input_vol)
        )
        assert output_vol.shape == input_vol.shape
        assert np.allclose(output_vol.voxel_size, input_vol.voxel_size)
        assert np.allclose(output_vol.corner, input_vol.corner)
        written = wa.open(str(tmp_path / "out.zarr" / "seg" / "s0")).translation
        assert np.allclose(written, translation)


class TestMultiDatasetConsistency:
    def test_read_onto_another_datasets_grid(self, tmp_path):
        """A 4 nm dataset read onto a 3.54 x 4 x 4 nm dataset's grid lines up with it."""
        shape = (5, 5, 5)
        ds1 = _create(str(tmp_path / "ds1.zarr" / "a"), shape, (3.54, 4.0, 4.0))
        ds1.write_index((0, 0, 0), shape, np.ones(shape, dtype=np.uint8))
        ds2 = _create(str(tmp_path / "ds2.zarr" / "b"), shape, (4.0, 4.0, 4.0))
        data2 = np.arange(125, dtype=np.uint8).reshape(shape)
        ds2.write_index((0, 0, 0), shape, data2)

        vol1 = open_dataset(str(tmp_path / "ds1.zarr" / "a" / "s0"))
        vol2 = open_dataset(str(tmp_path / "ds2.zarr" / "b" / "s0"))
        result = vol2.read_onto(grid_of(vol1))

        assert result.shape == shape
        # z: the 3.54 nm voxel centres fall in 4 nm voxels 0, 1, 2, 3, 3
        assert np.array_equal(result, data2[[0, 1, 2, 3, 3]])

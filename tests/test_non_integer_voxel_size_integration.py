"""Integration tests for non-integer voxel size support."""
import numpy as np
import pytest
import worldarray as wa
import zarr
from funlib.geometry import Coordinate, Roi

from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.zarr_util import (
    create_multiscale_dataset,
    create_multiscale_dataset_idi,
)
from cellmap_analyze.util.measure_util import get_region_properties
from cellmap_analyze.util.voxel_size_utils import scale_voxel_size_to_integers


@pytest.fixture
def non_int_zarr(tmp_path):
    """Create a zarr dataset with non-integer voxel_size in attributes."""
    zarr_path = str(tmp_path / "test.zarr")
    original_vs = (3.54, 4.0, 4.0)

    # Scale for funlib compatibility
    scaled_vs, scale_factor = scale_voxel_size_to_integers(original_vs)
    data_shape = (10, 10, 10)

    total_roi = Roi(
        Coordinate(0, 0, 0), Coordinate(data_shape) * Coordinate(scaled_vs)
    )

    ds = create_multiscale_dataset(
        f"{zarr_path}/test_ds",
        dtype=np.uint8,
        voxel_size=Coordinate(scaled_vs),
        total_roi=total_roi,
        write_size=Coordinate(data_shape) * Coordinate(scaled_vs),
        original_voxel_size=original_vs,
    )

    # Write test data: a simple labeled volume
    test_data = np.zeros(data_shape, dtype=np.uint8)
    test_data[2:5, 2:5, 2:5] = 1  # 27-voxel cube labeled 1
    test_data[6:8, 6:8, 6:8] = 2  # 8-voxel cube labeled 2
    ds.write_index((0, 0, 0), ds.shape, test_data)

    return zarr_path, original_vs, scaled_vs, scale_factor, test_data


class TestIDINonIntegerVoxelSize:
    def test_reads_original_voxel_size(self, non_int_zarr):
        zarr_path, original_vs, scaled_vs, scale_factor, _ = non_int_zarr
        idi = ImageDataInterface(f"{zarr_path}/test_ds/s0")

        assert idi.original_voxel_size == original_vs
        assert idi.voxel_size_scale_factor == scale_factor
        assert tuple(idi.voxel_size) == scaled_vs

    def test_voxel_counts_correct(self, non_int_zarr):
        """ROI / voxel_size should give correct voxel counts."""
        zarr_path, _, _, _, test_data = non_int_zarr
        idi = ImageDataInterface(f"{zarr_path}/test_ds/s0")

        voxel_shape = idi.roi.shape / idi.voxel_size
        assert tuple(voxel_shape) == test_data.shape

    def test_data_read_correct(self, non_int_zarr):
        """Data should be readable without corruption."""
        zarr_path, _, _, _, test_data = non_int_zarr
        idi = ImageDataInterface(f"{zarr_path}/test_ds/s0")

        data = idi.to_ndarray_ts()
        np.testing.assert_array_equal(data, test_data)

    def test_integer_voxel_size_no_scaling(self, tmp_path):
        """Integer voxel sizes should have scale_factor=1."""
        zarr_path = str(tmp_path / "int.zarr")
        vs = (8, 8, 8)
        data_shape = (5, 5, 5)

        total_roi = Roi((0, 0, 0), Coordinate(data_shape) * Coordinate(vs))
        ds = create_multiscale_dataset(
            f"{zarr_path}/test",
            dtype=np.uint8,
            voxel_size=Coordinate(vs),
            total_roi=total_roi,
            write_size=Coordinate(data_shape) * Coordinate(vs),
        )
        ds.write_index((0, 0, 0), ds.shape, np.ones(data_shape, dtype=np.uint8))

        idi = ImageDataInterface(f"{zarr_path}/test/s0")
        assert idi.voxel_size_scale_factor == 1
        assert tuple(idi.voxel_size) == vs
        assert idi.original_voxel_size == tuple(float(v) for v in vs)


class TestMeasurementsWithNonIntegerVoxelSize:
    def test_volume_correct(self, non_int_zarr):
        """Volume should use original voxel_size, not scaled."""
        zarr_path, original_vs, _, _, test_data = non_int_zarr
        idi = ImageDataInterface(f"{zarr_path}/test_ds/s0")

        # Compute region properties using original_voxel_size
        df = get_region_properties(
            test_data, voxel_size=idi.original_voxel_size, trim=0
        )

        true_voxel_volume = original_vs[0] * original_vs[1] * original_vs[2]
        # Label 1 has 27 voxels (3x3x3 cube)
        label_1_row = df[df["ID"] == 1].iloc[0]
        assert abs(label_1_row["Volume (nm^3)"] - 27 * true_voxel_volume) < 1e-6

        # Label 2 has 8 voxels (2x2x2 cube)
        label_2_row = df[df["ID"] == 2].iloc[0]
        assert abs(label_2_row["Volume (nm^3)"] - 8 * true_voxel_volume) < 1e-6

    def test_surface_area_correct(self, non_int_zarr):
        """Surface area should use original voxel_size face areas."""
        zarr_path, original_vs, _, _, test_data = non_int_zarr
        idi = ImageDataInterface(f"{zarr_path}/test_ds/s0")

        df = get_region_properties(
            test_data, voxel_size=idi.original_voxel_size, trim=0
        )

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
        scaled_vs, scale_factor = scale_voxel_size_to_integers(original_vs)

        total_roi = Roi(
            Coordinate(0, 0, 0), Coordinate(10, 10, 10) * Coordinate(scaled_vs)
        )

        create_multiscale_dataset(
            output_path,
            dtype=np.uint8,
            voxel_size=Coordinate(scaled_vs),
            total_roi=total_roi,
            write_size=Coordinate(10, 10, 10) * Coordinate(scaled_vs),
            original_voxel_size=original_vs,
        )

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
        original_vs = (3.54, 4.0, 4.0)
        scaled_vs, scale_factor = scale_voxel_size_to_integers(original_vs)

        total_roi = Roi(
            Coordinate(0, 0, 0), Coordinate(5, 5, 5) * Coordinate(scaled_vs)
        )

        create_multiscale_dataset(
            output_path,
            dtype=np.uint8,
            voxel_size=Coordinate(scaled_vs),
            total_roi=total_roi,
            write_size=Coordinate(5, 5, 5) * Coordinate(scaled_vs),
            original_voxel_size=original_vs,
        )

        assert dict(zarr.open_array(output_path + "/s0", mode="r").attrs) == {}

    def test_written_voxel_size_is_true_physical_value(self, tmp_path):
        """The persisted voxel size must be the TRUE physical voxel size, not
        the scaled integer, so external tools get correct units.
        cellmap-analyze must still re-derive the scaled integer in memory."""
        output_path = str(tmp_path / "true_vs.zarr" / "test_ds")
        original_vs = (72.96, 64.0, 64.0)
        scaled_vs, scale_factor = scale_voxel_size_to_integers(original_vs)
        assert scale_factor != 1  # ensure we're exercising the fractional path

        total_roi = Roi(
            Coordinate(0, 0, 0), Coordinate(5, 5, 5) * Coordinate(scaled_vs)
        )
        create_multiscale_dataset(
            output_path,
            dtype=np.uint8,
            voxel_size=Coordinate(scaled_vs),
            total_roi=total_roi,
            write_size=Coordinate(5, 5, 5) * Coordinate(scaled_vs),
            original_voxel_size=original_vs,
        )

        # What an external reader sees is the true value.
        written = wa.open(output_path + "/s0").voxel_size
        for s, e in zip(written, original_vs):
            assert abs(float(s) - e) < 1e-10, "voxel size should be physical"

        # cellmap-analyze still reconstructs the scaled-integer convention.
        idi = ImageDataInterface(output_path + "/s0")
        assert tuple(idi.original_voxel_size) == original_vs
        assert tuple(idi.voxel_size) == tuple(scaled_vs)
        assert idi.voxel_size_scale_factor == scale_factor

    @pytest.mark.parametrize(
        "original_vs, translation",
        [((5.0, 5.0, 5.0), (0.0, 0.0, 0.0)), ((8.0, 8.0, 8.0), (4.5, 0.0, 0.0))],
    )
    def test_output_keeps_input_placement(self, tmp_path, original_vs, translation):
        """An output written in its input's frame sits where the input does,
        when the frame's scale factor comes from the corner rather than the
        voxel size (here 2, for corners at -2.5 nm and 0.5 nm)."""
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
        input_idi = ImageDataInterface(str(tmp_path / "in.zarr" / "s0"))
        assert input_idi.voxel_size_scale_factor == 2

        output_idi = create_multiscale_dataset_idi(
            str(tmp_path / "out.zarr" / "seg"),
            dtype=np.uint8,
            voxel_size=input_idi.voxel_size,
            total_roi=input_idi.roi,
            write_size=input_idi.chunk_shape * input_idi.voxel_size,
            original_voxel_size=input_idi.original_voxel_size,
        )
        assert output_idi.voxel_size_scale_factor == input_idi.voxel_size_scale_factor
        assert output_idi.roi == input_idi.roi
        written = wa.open(str(tmp_path / "out.zarr" / "seg" / "s0")).translation
        assert np.allclose(written, translation)


class TestMultiDatasetConsistency:
    def test_rescale_to_common_factor(self, tmp_path):
        """Two IDIs with different scale factors should align correctly."""
        # Dataset 1: voxel_size (3.54, 4, 4) -> scale_factor 50
        zarr1 = str(tmp_path / "ds1.zarr")
        vs1 = (3.54, 4.0, 4.0)
        scaled_vs1, sf1 = scale_voxel_size_to_integers(vs1)
        shape = (5, 5, 5)
        roi1 = Roi(Coordinate(0, 0, 0), Coordinate(shape) * Coordinate(scaled_vs1))
        ds1 = create_multiscale_dataset(
            f"{zarr1}/a", np.uint8, Coordinate(scaled_vs1), roi1,
            write_size=Coordinate(shape) * Coordinate(scaled_vs1), original_voxel_size=vs1,
        )
        ds1.write_index((0, 0, 0), shape, np.ones(shape, dtype=np.uint8))

        # Dataset 2: voxel_size (4, 4, 4) -> scale_factor 1
        zarr2 = str(tmp_path / "ds2.zarr")
        vs2 = (4.0, 4.0, 4.0)
        scaled_vs2, sf2 = scale_voxel_size_to_integers(vs2)
        roi2 = Roi(Coordinate(0, 0, 0), Coordinate(shape) * Coordinate(scaled_vs2))
        ds2 = create_multiscale_dataset(
            f"{zarr2}/b", np.uint8, Coordinate(scaled_vs2), roi2,
            write_size=Coordinate(shape) * Coordinate(scaled_vs2), original_voxel_size=vs2,
        )
        ds2.write_index((0, 0, 0), shape, np.ones(shape, dtype=np.uint8))

        idi1 = ImageDataInterface(f"{zarr1}/a/s0")
        idi2 = ImageDataInterface(f"{zarr2}/b/s0")

        assert idi1.voxel_size_scale_factor == 50
        assert idi2.voxel_size_scale_factor == 1

        # Align to common factor
        from cellmap_analyze.util.voxel_size_utils import compute_common_scale_factor

        common_sf = compute_common_scale_factor(
            idi1.voxel_size_scale_factor, idi2.voxel_size_scale_factor
        )
        assert common_sf == 50

        idi1.rescale_to_factor(common_sf)
        idi2.rescale_to_factor(common_sf)

        # Both should now be in the same scaled space
        assert idi1.voxel_size_scale_factor == 50
        assert idi2.voxel_size_scale_factor == 50
        assert tuple(idi1.voxel_size) == (177, 200, 200)
        assert tuple(idi2.voxel_size) == (200, 200, 200)

        # Voxel counts should still be correct
        assert tuple(idi1.roi.shape / idi1.voxel_size) == shape
        assert tuple(idi2.roi.shape / idi2.voxel_size) == shape

import numpy as np
import pytest
from cellmap_analyze.util.image_data_interface import (
    ImageDataInterface,
)

from funlib.geometry import Roi, Coordinate


def test_image_data_interface_whole(
    tmp_zarr,
    image_with_holes_filled,
):
    test_data = ImageDataInterface(
        f"{tmp_zarr}/image_with_holes_filled/s0"
    ).to_ndarray_ts()
    assert np.array_equal(
        test_data,
        image_with_holes_filled,
    )


def test_image_data_interface_roi(tmp_zarr, image_with_holes_filled):
    idi = ImageDataInterface(f"{tmp_zarr}/image_with_holes_filled/s0")

    roi = Roi((1, 1, 1), (5, 5, 5))
    # Use IDI's voxel_size (scaled integers) for ROI operations
    test_data = idi.to_ndarray_ts(roi * idi.voxel_size)
    ground_truth = image_with_holes_filled[roi.to_slices()]
    assert np.array_equal(
        test_data,
        ground_truth,
    )


@pytest.mark.parametrize("fill_value", [0, 100])
def test_image_data_interface_constant_fill_values(
    tmp_zarr, image_with_holes_filled, fill_value
):
    idi = ImageDataInterface(
        f"{tmp_zarr}/image_with_holes_filled/s0", custom_fill_value=fill_value
    )

    roi = Roi((-1, -1, -1), (6, 6, 6))
    test_data = idi.to_ndarray_ts(roi * idi.voxel_size)
    ground_truth = np.pad(
        image_with_holes_filled[:5, :5, :5],
        pad_width=((1, 0), (1, 0), (1, 0)),
        mode="constant",
        constant_values=fill_value,
    )

    assert np.array_equal(
        test_data,
        ground_truth,
    )


def test_image_data_interface_reflect(tmp_zarr, image_with_holes_filled):
    idi = ImageDataInterface(
        f"{tmp_zarr}/image_with_holes_filled/s0", custom_fill_value="edge"
    )

    roi = Roi((-1, -1, -1), (6, 6, 6))
    test_data = idi.to_ndarray_ts(roi * idi.voxel_size)
    ground_truth = np.pad(
        image_with_holes_filled[:5, :5, :5],
        pad_width=((1, 0), (1, 0), (1, 0)),
        mode="edge",
    )

    assert np.array_equal(
        test_data,
        ground_truth,
    )


def test_nonzero_translation_center_corner_convention(tmp_path, voxel_size):
    """Regression for the OME voxel-CENTER convention with a NON-ZERO
    translation (the rest of the suite only exercises translation 0).

    The stored OME translation is the voxel CENTER of voxel [0,0,0]; internally
    the IDI exposes the funlib CORNER (= translation - vs/2). Writing converts
    corner -> center, so the stored translation round-trips, and measure reports
    COMs at true OME centers.
    """
    from cellmap_analyze.util.zarr_util import create_multiscale_dataset
    from cellmap_analyze.util.voxel_size_utils import scale_voxel_size_to_integers
    from cellmap_analyze.analyze.measure import Measure
    import zarr

    vs = tuple(float(v) for v in voxel_size)
    scaled_vs, sf = scale_voxel_size_to_integers(vs)
    # Pick a clean integer corner (5 voxels in) so there is no rounding slack;
    # the corresponding OME center is corner + vs/2 = 5.5 voxels.
    corner_scaled = tuple(5 * s for s in scaled_vs)
    expected_translation = tuple(c / sf + v / 2 for c, v in zip(corner_scaled, vs))

    shape = (8, 8, 8)
    total_roi = Roi(
        Coordinate(corner_scaled), Coordinate(shape) * Coordinate(scaled_vs)
    )
    path = str(tmp_path / "data.zarr/seg")
    ds = create_multiscale_dataset(
        path,
        dtype=np.uint8,
        voxel_size=list(scaled_vs),
        total_roi=total_roi,
        write_size=Coordinate(shape) * Coordinate(scaled_vs),
        original_voxel_size=list(vs),
    )
    idx = (1, 2, 3)
    arr = np.zeros(shape, dtype=np.uint8)
    arr[idx] = 1
    ds.data[:] = arr

    # stored OME translation is the CENTER
    zattrs = dict(zarr.open_group(path, mode="r").attrs)
    stored = zattrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][
        1
    ]["translation"]
    assert np.allclose(stored, expected_translation)

    # IDI exposes the CORNER = translation - vs/2 (round-trips the corner)
    idi = ImageDataInterface(f"{path}/s0")
    assert tuple(idi.offset) == corner_scaled

    # measure COM lands on the OME center of the object voxel: T + idx*vs
    m = Measure(input_path=f"{path}/s0", output_path=str(tmp_path / "csvs"), num_workers=1)
    m.measure()
    expected_com = tuple(expected_translation[d] + idx[d] * vs[d] for d in range(3))
    assert np.allclose(tuple(m.measurements[1].com), expected_com)


def test_raw_intensity_excludes_voxels_outside_raw_coverage(tmp_path):
    """An object whose segmentation extends past the raw EM dataset's own
    coverage must have its mean/std computed only from the voxels that
    actually have real EM data -- not have the missing region's padding
    (0) silently averaged in as if it were real signal.
    """
    from cellmap_analyze.util.zarr_io import prepare_ds
    from cellmap_analyze.analyze.measure import Measure

    voxel_size = Coordinate((10, 10, 10))

    # Segmentation: a single solid object filling a (10, 3, 3) block.
    seg_shape = (10, 3, 3)
    seg_data = np.ones(seg_shape, dtype=np.uint8)
    seg_total_roi = Roi((0, 0, 0), seg_shape) * voxel_size
    seg_path = str(tmp_path / "data.zarr/seg")
    seg_ds = prepare_ds(
        seg_path,
        "s0",
        total_roi=seg_total_roi,
        voxel_size=voxel_size,
        dtype=seg_data.dtype,
    )
    seg_ds[seg_total_roi] = seg_data

    # Raw EM only covers x in [0, 6) of the same voxel grid -- the object's
    # x in [6, 10) has no real EM data at all.
    raw_shape = (6, 3, 3)
    raw_value = 100
    raw_data = np.full(raw_shape, raw_value, dtype=np.uint16)
    raw_total_roi = Roi((0, 0, 0), raw_shape) * voxel_size
    raw_path = str(tmp_path / "data.zarr/raw")
    raw_ds = prepare_ds(
        raw_path,
        "s0",
        total_roi=raw_total_roi,
        voxel_size=voxel_size,
        dtype=raw_data.dtype,
    )
    raw_ds[raw_total_roi] = raw_data

    m = Measure(
        input_path=f"{seg_path}/s0",
        output_path=str(tmp_path / "csvs"),
        num_workers=1,
        raw_path=f"{raw_path}/s0",
    )
    m.measure()

    oi = m.measurements[1]
    covered_voxels = 6 * 3 * 3
    assert oi.raw_count == covered_voxels
    assert np.isclose(oi.mean_intensity, raw_value)
    assert np.isclose(oi.std_intensity, 0.0)

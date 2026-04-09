import pytest
from cellmap_analyze.analyze.measure import Measure
from cellmap_analyze.util.information_holders import ObjectInformation
from cellmap_analyze.util.measure_util import get_object_information, get_raw_intensity_stats

import numpy as np
import pandas as pd

from tests.test_utils import (
    simple_object_information_dict,
    simple_contact_site_information_dict,
    simple_raw_intensity_stats,
)


@pytest.fixture()
def contact_site_information_dict_contact_distance_8nm(
    segmentation_1, segmentation_2, contact_sites_distance_8nm, voxel_size
):
    return simple_contact_site_information_dict(
        segmentation_1, segmentation_2, contact_sites_distance_8nm, voxel_size
    )


@pytest.fixture()
def contact_site_information_dict_contact_distance_16nm(
    segmentation_1, segmentation_2, contact_sites_distance_16nm, voxel_size
):
    return simple_contact_site_information_dict(
        segmentation_1, segmentation_2, contact_sites_distance_16nm, voxel_size
    )


@pytest.fixture()
def contact_site_information_dict_contact_distance_24nm(
    segmentation_1, segmentation_2, contact_sites_distance_24nm, voxel_size
):
    return simple_contact_site_information_dict(
        segmentation_1, segmentation_2, contact_sites_distance_24nm, voxel_size
    )


def test_measure_whole_objects(
    segmentation_1, segmentation_2, segmentation_1_downsampled, voxel_size
):
    for segmentation, vs in [
        (segmentation_1, voxel_size),
        (segmentation_2, voxel_size),
        (segmentation_1_downsampled, voxel_size * 2),
    ]:
        object_information_dict = simple_object_information_dict(segmentation, vs)
        test_object_information_dict = get_object_information(
            segmentation, voxel_size=vs
        )
        assert object_information_dict == test_object_information_dict


def test_measure_blockwise_objects(
    shared_tmpdir,
    tmp_zarr,
    segmentation_1,
    segmentation_2,
    segmentation_1_downsampled,
    voxel_size,
):
    for segmentation, segmentation_name, vs in [
        (segmentation_1, "segmentation_1", voxel_size),
        (segmentation_2, "segmentation_2", voxel_size),
        (segmentation_1_downsampled, "segmentation_1_downsampled", voxel_size * 2),
    ]:
        object_information_dict = simple_object_information_dict(segmentation, vs)
        test_object_information_dict = get_object_information(
            segmentation, voxel_size=vs
        )
        assert object_information_dict == test_object_information_dict
        compare_measurements(
            shared_tmpdir,
            f"{tmp_zarr}/{segmentation_name}/s0",
            None,
            None,
            object_information_dict,
        )


def test_measure_whole_contact_sites_distance_8nm(
    segmentation_1,
    segmentation_2,
    voxel_size,
    contact_sites_distance_8nm,
    contact_site_information_dict_contact_distance_8nm,
):
    test_contact_site_information_dict = get_object_information(
        contact_sites_distance_8nm,
        organelle_1=segmentation_1,
        organelle_2=segmentation_2,
        voxel_size=voxel_size,
    )
    assert (
        contact_site_information_dict_contact_distance_8nm
        == test_contact_site_information_dict
    )


def test_measure_whole_contact_sites_distance_16nm(
    segmentation_1,
    segmentation_2,
    voxel_size,
    contact_sites_distance_16nm,
    contact_site_information_dict_contact_distance_16nm,
):
    test_contact_site_information_dict = get_object_information(
        contact_sites_distance_16nm,
        organelle_1=segmentation_1,
        organelle_2=segmentation_2,
        voxel_size=voxel_size,
    )

    assert (
        contact_site_information_dict_contact_distance_16nm
        == test_contact_site_information_dict
    )


def test_measure_whole_contact_sites_distance_24nm(
    segmentation_1,
    segmentation_2,
    voxel_size,
    contact_sites_distance_24nm,
    contact_site_information_dict_contact_distance_24nm,
):
    test_contact_site_information_dict = get_object_information(
        contact_sites_distance_24nm,
        organelle_1=segmentation_1,
        organelle_2=segmentation_2,
        voxel_size=voxel_size,
    )

    assert (
        contact_site_information_dict_contact_distance_24nm
        == test_contact_site_information_dict
    )


def compare_measurements(
    shared_tmpdir,
    input_ds_path,
    segmentation_1_path,
    segmentation_2_path,
    contact_site_information_dict,
):
    m = Measure(
        input_ds_path,
        organelle_1_path=segmentation_1_path,
        organelle_2_path=segmentation_2_path,
        output_path=shared_tmpdir,
        num_workers=1,
    )
    m.measure()
    assert m.measurements == contact_site_information_dict


def test_writeout(shared_tmpdir, tmp_zarr, tmp_cylinders_information_csv):
    m = Measure(
        input_path=f"{tmp_zarr}/segmentation_cylinders/s0",
        output_path=f"{shared_tmpdir}/test_csvs",
        num_workers=1,
    )
    m.get_measurements()
    gt_df = pd.read_csv(tmp_cylinders_information_csv)
    test_df = pd.read_csv(f"{shared_tmpdir}/test_csvs/segmentation_cylinders.csv")
    # Use assert_frame_equal with check_dtype=False since CSV serialization may change dtypes
    # (e.g., int64 -> float64) but values should be identical
    pd.testing.assert_frame_equal(gt_df, test_df, check_dtype=False)


@pytest.mark.parametrize(
    "segmentation_name", ["segmentation_1", "segmentation_2", "segmentation_random"]
)
def test_measure_blockwise(shared_tmpdir, tmp_zarr, segmentation_name, request):
    m = Measure(
        input_path=f"{tmp_zarr}/{segmentation_name}/s0",
        output_path=shared_tmpdir,
        num_workers=1,
    )
    m.measure()
    assert m.measurements == simple_object_information_dict(
        request.getfixturevalue(segmentation_name), m.input_idi.original_voxel_size
    )


def test_measure_blockwise_contact_sites_distance_8nm(
    shared_tmpdir, tmp_zarr, contact_site_information_dict_contact_distance_8nm
):
    compare_measurements(
        shared_tmpdir,
        f"{tmp_zarr}/contact_sites_distance_8nm/s0",
        f"{tmp_zarr}/segmentation_1/s0",
        f"{tmp_zarr}/segmentation_2/s0",
        contact_site_information_dict_contact_distance_8nm,
    )


def test_measure_blockwise_contact_sites_distance_16nm(
    shared_tmpdir, tmp_zarr, contact_site_information_dict_contact_distance_16nm
):
    compare_measurements(
        shared_tmpdir,
        f"{tmp_zarr}/contact_sites_distance_16nm/s0",
        f"{tmp_zarr}/segmentation_1/s0",
        f"{tmp_zarr}/segmentation_2/s0",
        contact_site_information_dict_contact_distance_16nm,
    )


def test_measure_blockwise_contact_sites_distance_24nm(
    shared_tmpdir, tmp_zarr, contact_site_information_dict_contact_distance_24nm
):
    compare_measurements(
        shared_tmpdir,
        f"{tmp_zarr}/contact_sites_distance_24nm/s0",
        f"{tmp_zarr}/segmentation_1/s0",
        f"{tmp_zarr}/segmentation_2/s0",
        contact_site_information_dict_contact_distance_24nm,
    )


def test_measure_blockwise_downsampled_contact_sites_distance_8nm(
    shared_tmpdir, tmp_zarr, contact_site_information_dict_contact_distance_8nm
):
    compare_measurements(
        shared_tmpdir,
        f"{tmp_zarr}/contact_sites_distance_8nm/s0",
        f"{tmp_zarr}/segmentation_1_downsampled/s0",
        f"{tmp_zarr}/segmentation_2/s0",
        contact_site_information_dict_contact_distance_8nm,
    )


def test_measure_blockwise_downsampled_contact_sites_distance_16nm(
    shared_tmpdir, tmp_zarr, contact_site_information_dict_contact_distance_16nm
):
    compare_measurements(
        shared_tmpdir,
        f"{tmp_zarr}/contact_sites_distance_16nm/s0",
        f"{tmp_zarr}/segmentation_1_downsampled/s0",
        f"{tmp_zarr}/segmentation_2/s0",
        contact_site_information_dict_contact_distance_16nm,
    )


def test_measure_blockwise_downsampled_contact_sites_distance_24nm(
    shared_tmpdir, tmp_zarr, contact_site_information_dict_contact_distance_24nm
):
    compare_measurements(
        shared_tmpdir,
        f"{tmp_zarr}/contact_sites_distance_24nm/s0",
        f"{tmp_zarr}/segmentation_1_downsampled/s0",
        f"{tmp_zarr}/segmentation_2/s0",
        contact_site_information_dict_contact_distance_24nm,
    )


def test_get_raw_intensity_stats(segmentation_2):
    np.random.seed(99)
    raw = np.random.rand(*segmentation_2.shape).astype(np.float32) * 100
    result = get_raw_intensity_stats(segmentation_2, raw, trim=0)
    gt = simple_raw_intensity_stats(segmentation_2, raw)
    for label_id in gt:
        assert label_id in result
        raw_sum, raw_sum_sq, raw_count = result[label_id]
        assert raw_count == gt[label_id]["count"]
        assert np.allclose(raw_sum, gt[label_id]["sum"], rtol=1e-5)
        assert np.allclose(raw_sum_sq, gt[label_id]["sum_sq"], rtol=1e-5)


def test_object_information_raw_intensity_add():
    oi1 = ObjectInformation(
        counts=10,
        volume=80,
        com=np.array([4.0, 4.0, 4.0]),
        surface_area=100,
        sum_r2=1000,
        bounding_box=[0, 0, 0, 8, 8, 8],
        raw_sum=500.0,
        raw_sum_sq=30000.0,
        raw_count=10,
    )
    oi2 = ObjectInformation(
        counts=5,
        volume=40,
        com=np.array([12.0, 12.0, 12.0]),
        surface_area=60,
        sum_r2=3000,
        bounding_box=[8, 8, 8, 16, 16, 16],
        raw_sum=250.0,
        raw_sum_sq=15000.0,
        raw_count=5,
    )
    merged = oi1 + oi2
    assert merged.has_raw_intensity
    assert merged.raw_sum == 750.0
    assert merged.raw_sum_sq == 45000.0
    assert merged.raw_count == 15
    assert np.allclose(merged.mean_intensity, 750.0 / 15)
    expected_std = np.sqrt(max(0, 45000.0 / 15 - (750.0 / 15) ** 2))
    assert np.allclose(merged.std_intensity, expected_std)


def test_measure_whole_objects_with_raw(segmentation_2, raw_intensity_for_seg2, voxel_size):
    object_information_dict = get_object_information(
        segmentation_2,
        voxel_size=voxel_size,
        raw_data=raw_intensity_for_seg2,
    )
    gt = simple_raw_intensity_stats(segmentation_2, raw_intensity_for_seg2)
    for label_id, oi in object_information_dict.items():
        assert oi.has_raw_intensity
        assert np.allclose(oi.mean_intensity, gt[label_id]["mean"], rtol=1e-5)
        assert np.allclose(oi.std_intensity, gt[label_id]["std"], rtol=1e-5)


def test_measure_blockwise_with_raw(
    shared_tmpdir, tmp_zarr, segmentation_2, raw_intensity_for_seg2, voxel_size
):
    m = Measure(
        input_path=f"{tmp_zarr}/segmentation_2/s0",
        output_path=shared_tmpdir,
        raw_path=f"{tmp_zarr}/raw_intensity_for_seg2/s0",
        num_workers=1,
    )
    m.measure()
    gt = simple_raw_intensity_stats(segmentation_2, raw_intensity_for_seg2)
    for label_id, oi in m.measurements.items():
        assert oi.has_raw_intensity
        assert np.allclose(oi.mean_intensity, gt[label_id]["mean"], rtol=1e-5)
        assert np.allclose(oi.std_intensity, gt[label_id]["std"], rtol=1e-5)

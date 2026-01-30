import numpy as np
import pytest
from cellmap_analyze.util.xarray_image_data_interface import (
    XarrayImageDataInterface,
)

from funlib.geometry import Roi


def test_image_data_interface_whole(
    tmp_zarr,
    image_with_holes_filled,
):
    test_data = XarrayImageDataInterface(
        f"{tmp_zarr}/image_with_holes_filled/s0"
    ).to_ndarray_ts()
    assert np.array_equal(
        test_data,
        image_with_holes_filled,
    )


def test_image_data_interface_roi(tmp_zarr, voxel_size, image_with_holes_filled):
    idi = XarrayImageDataInterface(f"{tmp_zarr}/image_with_holes_filled/s0")

    roi = Roi((1, 1, 1), (5, 5, 5))
    test_data = idi.to_ndarray_ts(roi * voxel_size)
    ground_truth = image_with_holes_filled[roi.to_slices()]
    assert np.array_equal(
        test_data,
        ground_truth,
    )


@pytest.mark.parametrize("fill_value", [0, 100])
def test_image_data_interface_constant_fill_values(
    tmp_zarr, voxel_size, image_with_holes_filled, fill_value
):
    idi = XarrayImageDataInterface(
        f"{tmp_zarr}/image_with_holes_filled/s0", custom_fill_value=fill_value
    )

    roi = Roi((-1, -1, -1), (6, 6, 6))
    test_data = idi.to_ndarray_ts(roi * voxel_size)
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


def test_image_data_interface_reflect(tmp_zarr, voxel_size, image_with_holes_filled):
    idi = XarrayImageDataInterface(
        f"{tmp_zarr}/image_with_holes_filled/s0", custom_fill_value="edge"
    )

    roi = Roi((-1, -1, -1), (6, 6, 6))
    test_data = idi.to_ndarray_ts(roi * voxel_size)
    ground_truth = np.pad(
        image_with_holes_filled[:5, :5, :5],
        pad_width=((1, 0), (1, 0), (1, 0)),
        mode="edge",
    )

    assert np.array_equal(
        test_data,
        ground_truth,
    )

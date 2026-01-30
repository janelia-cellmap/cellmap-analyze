from cellmap_analyze.process.fill_holes import FillHoles
import pytest
import numpy as np

from cellmap_analyze.util.xarray_image_data_interface import (
    XarrayImageDataInterface,
)


@pytest.mark.parametrize(
    "image_with_hole",
    ["image_with_holes", "tic_tac_toe"],
)
def test_fill_holes(tmp_zarr, image_with_hole):
    fh = FillHoles(
        input_path=f"{tmp_zarr}/{image_with_hole}/s0",
        output_path=f"{tmp_zarr}/{image_with_hole}_filled_test",
        num_workers=1,
        connectivity=1,
    )
    fh.fill_holes()

    ground_truth = XarrayImageDataInterface(
        f"{tmp_zarr}/{image_with_hole}_filled/s0"
    ).to_ndarray_ts()
    test_data = XarrayImageDataInterface(
        f"{tmp_zarr}/{image_with_hole}_filled_test/s0"
    ).to_ndarray_ts()
    assert np.array_equal(
        test_data,
        ground_truth,
    )

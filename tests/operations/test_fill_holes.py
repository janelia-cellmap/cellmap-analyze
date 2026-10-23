from cellmap_analyze.process.fill_holes import FillHoles
import pytest
import numpy as np

from tests.test_utils import read_all


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

    ground_truth = read_all(f"{tmp_zarr}/{image_with_hole}_filled/s0")
    test_data = read_all(f"{tmp_zarr}/{image_with_hole}_filled_test/s0")
    assert np.array_equal(
        test_data,
        ground_truth,
    )

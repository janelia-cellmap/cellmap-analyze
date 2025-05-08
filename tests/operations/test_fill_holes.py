from cellmap_analyze.process.fill_holes import FillHoles

import numpy as np

from cellmap_analyze.util.image_data_interface import (
    ImageDataInterface,
)


def test_fill_holes(
    tmp_zarr,
    image_with_holes_filled,
):
    fh = FillHoles(
        input_path=f"{tmp_zarr}/image_with_holes/s0",
        output_path=f"{tmp_zarr}/test_fill_holes",
        num_workers=1,
        connectivity=1,
    )
    fh.fill_holes()

    ground_truth = image_with_holes_filled
    test_data = ImageDataInterface(f"{tmp_zarr}/test_fill_holes/s0").to_ndarray_ts()
    assert np.array_equal(
        test_data,
        ground_truth,
    )

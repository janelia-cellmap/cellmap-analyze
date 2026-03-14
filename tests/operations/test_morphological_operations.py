import pytest
from cellmap_analyze.process.morphological_operations import MorphologicalOperations
import fastmorph
import numpy as np
from cellmap_analyze.util.image_data_interface import (
    ImageDataInterface,
)


@pytest.mark.parametrize("operation", ["erosion", "dilation"])
def test_morphological_operations(tmp_zarr, segmentation_cylinders, operation):
    num_iterations = 1
    mo = MorphologicalOperations(
        input_path=f"{tmp_zarr}/segmentation_cylinders/s0",
        output_path=f"{tmp_zarr}/test_morphological_{operation}",
        num_workers=1,
        operation=operation,
        iterations=num_iterations,
    )
    mo.perform_morphological_operation()
    test_data = ImageDataInterface(
        f"{tmp_zarr}/test_morphological_{operation}/s0"
    ).to_ndarray_ts()
    ground_truth = segmentation_cylinders.copy()

    if operation == "erosion":
        ground_truth = fastmorph.erode(ground_truth, iterations=num_iterations)
    else:
        ground_truth = fastmorph.dilate(ground_truth, iterations=num_iterations)

    assert np.array_equal(test_data, ground_truth)

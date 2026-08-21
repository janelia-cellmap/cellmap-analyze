import pytest
from cellmap_analyze.process.morphological_operations import MorphologicalOperations
import fastmorph
import numpy as np
from cellmap_analyze.util.image_data_interface import (
    ImageDataInterface,
)


def _reference_morphology(ground_truth, operation, num_iterations):
    """Direct (non-blockwise) fastmorph reference, in the same multilabel
    mode MorphologicalOperations uses -- the blockwise output must match
    this exactly, per the halo argument in the class docstring."""
    if operation == "erosion":
        return fastmorph.erode(ground_truth, iterations=num_iterations)
    if operation == "dilation":
        return fastmorph.dilate(ground_truth, iterations=num_iterations)
    if operation == "opening":
        eroded = fastmorph.erode(ground_truth, iterations=num_iterations)
        return fastmorph.dilate(eroded, iterations=num_iterations)
    if operation == "closing":
        dilated = fastmorph.dilate(ground_truth, iterations=num_iterations)
        return fastmorph.erode(dilated, iterations=num_iterations)
    raise ValueError(operation)


@pytest.mark.parametrize("num_iterations", [1, 2, 3])
@pytest.mark.parametrize("operation", ["erosion", "dilation", "opening", "closing"])
def test_morphological_operations(
    tmp_zarr, segmentation_cylinders, operation, num_iterations
):
    # segmentation_cylinders has 3 distinct instance ids plus a spurious
    # isolated single-voxel fleck (id 1 at [0, 0, 0]) -- exercising both
    # multi-instance safety (no id should bleed into another) and opening's
    # noise-stripping behavior. tmp_zarr's chunk size is small relative to
    # the (50, 50, 50) array, so this dispatches across many blocks, and the
    # session-default voxel_size is anisotropic -- together these exercise
    # the blockwise halo math this feature depends on for correctness, not
    # just a single in-memory whole-array call.
    mo = MorphologicalOperations(
        input_path=f"{tmp_zarr}/segmentation_cylinders/s0",
        output_path=f"{tmp_zarr}/test_morphological_{operation}_{num_iterations}",
        num_workers=1,
        operation=operation,
        iterations=num_iterations,
    )
    mo.perform_morphological_operation()
    test_data = ImageDataInterface(
        f"{tmp_zarr}/test_morphological_{operation}_{num_iterations}/s0"
    ).to_ndarray_ts()

    ground_truth = _reference_morphology(
        segmentation_cylinders.copy(), operation, num_iterations
    )

    assert np.array_equal(test_data, ground_truth)


def test_morphological_operations_invalid_operation(tmp_zarr, segmentation_cylinders):
    with pytest.raises(ValueError):
        MorphologicalOperations(
            input_path=f"{tmp_zarr}/segmentation_cylinders/s0",
            output_path=f"{tmp_zarr}/test_morphological_invalid",
            num_workers=1,
            operation="not_a_real_operation",
        )

import pytest
from cellmap_analyze.process.label_with_mask import LabelWithMask
import fastmorph
import numpy as np

from cellmap_analyze.util.xarray_image_data_interface import (
    XarrayImageDataInterface,
)


def ground_truth_label_with_mask(
    input,
    label_mask,
    intensity_threshold_minimum,
    intensity_threshold_maximum,
    surface_voxels_only,
):
    if surface_voxels_only:
        label_mask = label_mask - fastmorph.erode(
            label_mask, erode_border=False
        )  # NOTE: fastmorph assumes 3 connected structuring element, which we do not use, but works in this particular test case
    output = label_mask * (
        (input >= intensity_threshold_minimum) & (input < intensity_threshold_maximum)
    )
    return output


@pytest.mark.parametrize(
    "intensity_threshold_minimum, intensity_threshold_maximum, surface_voxels_only",
    [
        (1, np.inf, False),
        (1, np.inf, True),
        (2, 3, False),
    ],
)
def test_label_with_mask(
    tmp_zarr,
    image_with_holes_filled,
    label_mask,
    intensity_threshold_minimum,
    intensity_threshold_maximum,
    surface_voxels_only,
):
    output_name = f"test_label_mask_{intensity_threshold_minimum}_{intensity_threshold_maximum}_{surface_voxels_only}"
    lwm = LabelWithMask(
        input_path=f"{tmp_zarr}/image_with_holes_filled/s0",
        mask_path=f"{tmp_zarr}/label_mask/s0",
        output_path=f"{tmp_zarr}/{output_name}",
        num_workers=1,
        intensity_threshold_minimum=intensity_threshold_minimum,
        intensity_threshold_maximum=intensity_threshold_maximum,
        surface_voxels_only=surface_voxels_only,
    )
    lwm.get_label_with_mask()

    ground_truth = ground_truth_label_with_mask(
        image_with_holes_filled,
        label_mask,
        intensity_threshold_minimum,
        intensity_threshold_maximum,
        surface_voxels_only,
    )
    test_data = XarrayImageDataInterface(f"{tmp_zarr}/{output_name}/s0").to_ndarray_ts()
    assert np.array_equal(
        test_data,
        ground_truth,
    )

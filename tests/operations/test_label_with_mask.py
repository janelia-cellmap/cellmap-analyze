import pytest
from cellmap_analyze.process.label_with_mask import LabelWithMask

import numpy as np

from cellmap_analyze.util.image_data_interface import (
    ImageDataInterface,
)


def ground_truth_label_with_mask(
    input,
    label_mask,
    intensity_threshold_minimum,
    intensity_threshold_maximum,
):
    # Create a mask based on the intensity thresholds
    output = label_mask * (
        (input >= intensity_threshold_minimum) & (input < intensity_threshold_maximum)
    )
    return output


@pytest.mark.parametrize(
    "intensity_threshold_minimum, intensity_threshold_maximum",
    [
        (1, np.inf),
        (2, 3),
    ],
)
def test_label_with_mask(
    tmp_zarr,
    image_with_holes_filled,
    label_mask,
    intensity_threshold_minimum,
    intensity_threshold_maximum,
):
    lwm = LabelWithMask(
        input_path=f"{tmp_zarr}/image_with_holes_filled/s0",
        mask_path=f"{tmp_zarr}/label_mask/s0",
        output_path=f"{tmp_zarr}/test_label_mask_{intensity_threshold_minimum}_{intensity_threshold_maximum}",
        num_workers=1,
        intensity_threshold_minimum=intensity_threshold_minimum,
        intensity_threshold_maximum=intensity_threshold_maximum,
    )
    lwm.get_label_with_mask()

    ground_truth = ground_truth_label_with_mask(
        image_with_holes_filled,
        label_mask,
        intensity_threshold_minimum,
        intensity_threshold_maximum,
    )
    test_data = ImageDataInterface(
        f"{tmp_zarr}/test_label_mask_{intensity_threshold_minimum}_{intensity_threshold_maximum}/s0"
    ).to_ndarray_ts()
    assert np.array_equal(
        test_data,
        ground_truth,
    )

import pytest
from funlib.geometry import Coordinate
from scipy import ndimage
from cellmap_analyze.util.block_util import erosion, dilation
from cellmap_analyze.util.mask_util import MasksFromConfig

import numpy as np

from cellmap_analyze.util.image_data_interface import (
    ImageDataInterface,
)


def test_masks(tmp_zarr, mask_one, mask_two, voxel_size):
    iterations = 2
    mask_dict = {
        "mask_one": {
            "path": f"{tmp_zarr}/mask_one/s0",
            "operation": "erosion",
            "mask_type": "exclusive",
            "iterations": iterations,
        },
        "mask_two": {
            "path": f"{tmp_zarr}/mask_two/s0",
            "mask_type": "exclusive",
            "operation": "dilation",
            "iterations": iterations,
        },
    }

    connectivity = 2
    structuring_element = ndimage.generate_binary_structure(3, connectivity)
    mask_one = np.pad(mask_one, pad_width=[[iterations, iterations]] * 3, mode="edge")
    s = slice(iterations, -iterations)
    mask_one = erosion(mask_one, iterations, structuring_element).astype(np.uint8)[
        (s, s, s)
    ]
    ground_truth = (1 - mask_one) & (
        1 - dilation(mask_two, 2, structuring_element).astype(np.uint8)
    )

    test_masks = MasksFromConfig(
        mask_dict,
        output_voxel_size=Coordinate(3 * [voxel_size]),
        connectivity=connectivity,
    )
    test_data = test_masks.process_block(roi=None)
    assert np.array_equal(
        test_data,
        ground_truth,
    )

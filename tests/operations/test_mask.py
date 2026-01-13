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

    # Ensure voxel_size is a tuple (handle both scalar and array cases)
    if np.isscalar(voxel_size):
        output_voxel_size = Coordinate(3 * [voxel_size])
        voxel_size_tuple = (voxel_size, voxel_size, voxel_size)
    else:
        output_voxel_size = Coordinate(voxel_size)
        voxel_size_tuple = tuple(voxel_size)

    # Calculate per-axis padding in voxels for anisotropic data
    min_voxel_size = min(voxel_size_tuple)
    padding_nm = iterations * min_voxel_size
    padding_voxels_per_axis = tuple(
        int(np.round(padding_nm / vs)) for vs in voxel_size_tuple
    )

    mask_one = np.pad(
        mask_one,
        pad_width=[[p, p] for p in padding_voxels_per_axis],
        mode="edge"
    )
    slices = tuple(
        slice(p, -p) if p > 0 else slice(None)
        for p in padding_voxels_per_axis
    )
    mask_one = erosion(mask_one, iterations, structuring_element).astype(np.uint8)[slices]
    ground_truth = (1 - mask_one) & (
        1 - dilation(mask_two, 2, structuring_element).astype(np.uint8)
    )

    test_masks = MasksFromConfig(
        mask_dict,
        output_voxel_size=output_voxel_size,
        connectivity=connectivity,
    )
    test_data = test_masks.process_block(roi=None)
    assert np.array_equal(
        test_data,
        ground_truth,
    )

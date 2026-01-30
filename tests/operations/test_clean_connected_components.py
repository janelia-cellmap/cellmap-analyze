import pytest
from scipy import ndimage
from skimage import measure
from cellmap_analyze.process.clean_connected_components import CleanConnectedComponents

import numpy as np

from cellmap_analyze.util.xarray_image_data_interface import (
    XarrayImageDataInterface,
)


@pytest.mark.parametrize(
    "minimum_volume_voxels, maximum_volume_voxels, use_mask",
    [
        (0, 10, False),
        (9, np.inf, False),
        (0, 10, True),
        (9, np.inf, True),
    ],
)
def test_clean_connected_components(
    tmp_zarr,
    connected_components,
    minimum_volume_voxels,
    maximum_volume_voxels,
    use_mask,
    mask_one,
    voxel_size,
):
    mask_config = None
    if use_mask:
        mask_config = {
            "mask_one": {"path": f"{tmp_zarr}/mask_one/s0", "mask_type": "inclusive"}
        }
    minimum_volume_nm_3 = minimum_volume_voxels * voxel_size**3
    maximum_volume_nm_3 = maximum_volume_voxels * voxel_size**3
    ccc = CleanConnectedComponents(
        input_path=f"{tmp_zarr}/connected_components/s0",
        output_path=f"{tmp_zarr}/test_clean_connected_components_minimum_volume_nm_3_{minimum_volume_nm_3}_maximum_volume_nm_3_{maximum_volume_nm_3}_{use_mask}",
        minimum_volume_nm_3=minimum_volume_nm_3,
        maximum_volume_nm_3=maximum_volume_nm_3,
        mask_config=mask_config,
        num_workers=1,
        connectivity=1,
    )
    ccc.clean_connected_components()

    ground_truth = connected_components.copy()
    if use_mask:
        ground_truth *= mask_one > 0

    uniques, counts = np.unique(ground_truth, return_counts=True)
    relabeling_dict = dict(zip(uniques, [0] * len(uniques)))
    counts = counts[uniques > 0]
    uniques = uniques[uniques > 0]
    uniques = uniques[
        (counts >= minimum_volume_voxels) & (counts < maximum_volume_voxels)
    ]
    relabeling_dict.update({k: i + 1 for i, k in enumerate(uniques)})
    ground_truth = np.vectorize(relabeling_dict.get)(ground_truth)
    test_data = XarrayImageDataInterface(
        f"{tmp_zarr}/test_clean_connected_components_minimum_volume_nm_3_{minimum_volume_nm_3}_maximum_volume_nm_3_{maximum_volume_nm_3}_{use_mask}/s0"
    ).to_ndarray_ts()
    assert np.array_equal(
        test_data,
        ground_truth,
    )

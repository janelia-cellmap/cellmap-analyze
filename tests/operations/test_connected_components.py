import pytest
from cellmap_analyze.analyze.measure import Measure
from cellmap_analyze.process.connected_components import ConnectedComponents

import numpy as np

from cellmap_analyze.util.image_data_interface import (
    ImageDataInterface,
)


@pytest.mark.parametrize("minimum_volume_voxels", [0, 1, 7, 9, 64, 65])
def test_connected_components(
    tmp_zarr,
    connected_components,
    minimum_volume_voxels,
    voxel_size,
):
    minimum_volume_nm_3 = minimum_volume_voxels * voxel_size**3
    cc = ConnectedComponents(
        tmp_blockwise_ds_path=f"{tmp_zarr}/blockwise_connected_components/s0",
        output_ds_path=f"{tmp_zarr}/test_connected_components_minimum_volume_nm_3_{minimum_volume_nm_3}",
        minimum_volume_nm_3=minimum_volume_nm_3,
        num_workers=1,
        connectivity=3,
    )
    cc.merge_connected_components_across_blocks()

    ground_truth = connected_components
    uniques, counts = np.unique(ground_truth, return_counts=True)
    relabeling_dict = dict(zip(uniques, [0] * len(uniques)))
    counts = counts[uniques > 0]
    uniques = uniques[uniques > 0]
    uniques = uniques[counts >= minimum_volume_voxels]
    relabeling_dict.update({k: i + 1 for i, k in enumerate(uniques)})
    ground_truth = np.vectorize(relabeling_dict.get)(ground_truth)
    test_data = ImageDataInterface(
        f"{tmp_zarr}/test_connected_components_minimum_volume_nm_3_{minimum_volume_nm_3}/s0"
    ).to_ndarray_ts()
    assert np.array_equal(
        test_data,
        ground_truth,
    )

import pytest
from scipy import ndimage
from skimage import measure
from cellmap_analyze.process.connected_components import ConnectedComponents

import numpy as np

from cellmap_analyze.util.image_data_interface import (
    ImageDataInterface,
)


@pytest.mark.parametrize(
    "minimum_volume_voxels, maximum_volume_voxels",
    [
        (0, np.inf),
        (1, np.inf),
        (7, np.inf),
        (9, np.inf),
        (64, np.inf),
        (65, np.inf),
        (4, 63),
    ],
)
def test_connected_components(
    tmp_zarr,
    connected_components,
    minimum_volume_voxels,
    maximum_volume_voxels,
    voxel_size,
):
    minimum_volume_nm_3 = minimum_volume_voxels * voxel_size**3
    maximum_volume_nm_3 = maximum_volume_voxels * voxel_size**3
    cc = ConnectedComponents(
        input_path=f"{tmp_zarr}/intensity_image/s0",
        output_path=f"{tmp_zarr}/test_connected_components_minimum_volume_nm_3_{minimum_volume_nm_3}_maximum_volume_nm_3_{maximum_volume_nm_3}",
        intensity_threshold_minimum=127,
        minimum_volume_nm_3=minimum_volume_nm_3,
        maximum_volume_nm_3=maximum_volume_nm_3,
        num_workers=1,
        connectivity=1,
    )
    cc.get_connected_components()

    ground_truth = connected_components.copy()
    uniques, counts = np.unique(ground_truth, return_counts=True)
    relabeling_dict = dict(zip(uniques, [0] * len(uniques)))
    counts = counts[uniques > 0]
    uniques = uniques[uniques > 0]
    uniques = uniques[
        (counts >= minimum_volume_voxels) & (counts < maximum_volume_voxels)
    ]
    relabeling_dict.update({k: i + 1 for i, k in enumerate(uniques)})
    ground_truth = np.vectorize(relabeling_dict.get)(ground_truth)
    test_data = ImageDataInterface(
        f"{tmp_zarr}/test_connected_components_minimum_volume_nm_3_{minimum_volume_nm_3}_maximum_volume_nm_3_{maximum_volume_nm_3}/s0"
    ).to_ndarray_ts()
    assert np.array_equal(
        test_data,
        ground_truth,
    )


def test_connected_components_filled(
    tmp_zarr,
    image_with_holes,
):
    cc = ConnectedComponents(
        input_path=f"{tmp_zarr}/image_with_holes/s0",
        output_path=f"{tmp_zarr}/test_connected_components_hole_filling",
        intensity_threshold_minimum=1,
        num_workers=1,
        connectivity=1,
        fill_holes=True,
    )
    cc.get_connected_components()

    test_data = ImageDataInterface(
        f"{tmp_zarr}/test_connected_components_hole_filling_filled/s0"
    ).to_ndarray_ts()

    ground_truth_filled = measure.label(image_with_holes > 0, connectivity=1).astype(
        np.uint64
    )
    ground_truth = np.zeros_like(ground_truth_filled)
    for id in np.unique(ground_truth_filled[ground_truth_filled > 0]):
        ground_truth += (
            ndimage.binary_fill_holes(
                ground_truth_filled == id, ndimage.generate_binary_structure(3, 1)
            ).astype(np.uint64)
            * id
        )

    assert np.array_equal(
        test_data,
        ground_truth,
    )


def test_connected_components_chunk_shape(
    tmp_zarr,
    image_with_holes,
):
    original_chunk_shape = ImageDataInterface(
        f"{tmp_zarr}/test_connected_components_hole_filling_filled/s0"
    ).chunk_shape
    cc = ConnectedComponents(
        input_path=f"{tmp_zarr}/image_with_holes/s0",
        output_path=f"{tmp_zarr}/test_connected_components_hole_filling",
        intensity_threshold_minimum=1,
        num_workers=1,
        connectivity=1,
        fill_holes=True,
        chunk_shape=list(original_chunk_shape * 2),
    )
    cc.get_connected_components()
    test_data_idi = ImageDataInterface(
        f"{tmp_zarr}/test_connected_components_hole_filling_filled/s0"
    )
    test_data = test_data_idi.to_ndarray_ts()

    ground_truth_filled = measure.label(image_with_holes > 0, connectivity=1).astype(
        np.uint64
    )
    ground_truth = np.zeros_like(ground_truth_filled)
    for id in np.unique(ground_truth_filled[ground_truth_filled > 0]):
        ground_truth += (
            ndimage.binary_fill_holes(
                ground_truth_filled == id, ndimage.generate_binary_structure(3, 1)
            ).astype(np.uint64)
            * id
        )
    assert (
        np.array_equal(
            test_data,
            ground_truth,
        )
        and test_data_idi.chunk_shape == original_chunk_shape * 2
    )

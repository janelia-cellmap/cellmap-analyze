import pytest
from cellmap_analyze.process.flawed_watershed_segmentation import FlawedWatershedSegmentation
from tests.test_utils import arrays_equal_up_to_id_ordering
import edt
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from cellmap_analyze.util.image_data_interface import (
    ImageDataInterface,
)
import fastremap
import cc3d


def global_watershed(input, pseudo_neighborhood_radius_voxels, voxel_size):
    global_distance = edt.edt(input, black_border=True) * voxel_size
    global_coords = peak_local_max(
        global_distance,
        footprint=np.ones((2 * pseudo_neighborhood_radius_voxels + 1,) * 3),
        labels=input,
        exclude_border=False,
    )
    global_seeds = np.zeros_like(global_distance, dtype=np.uint64)
    global_seeds[tuple(global_coords.T)] = 1
    global_seeds[global_seeds > 0] += input[global_seeds > 0]
    global_seeds = cc3d.connected_components(global_seeds, connectivity=26)
    global_watershed = np.zeros_like(global_distance, dtype=np.uint32)
    for id in fastremap.unique(input):
        if id == 0:
            pass
        mask = input == id
        distance_transform_masked = global_distance * mask
        watershed_seeds_masked = global_seeds * mask
        global_watershed += watershed(
            -distance_transform_masked,
            markers=watershed_seeds_masked,
            mask=distance_transform_masked > 0,
            connectivity=1,
        )
    return global_watershed


@pytest.mark.parametrize(
    "fixture_name, image_name",
    [
        ("image_with_holes_filled", "image_with_holes_filled"),
        ("segmentation_spheres", "segmentation_spheres"),
    ],
    ids=["image_with_holes_filled", "segmentation_spheres"],
)
@pytest.mark.parametrize(
    "pseudo_neighborhood_radius_voxels",
    [2, 3],
    ids=["radius_2voxels", "radius_3voxels"],
)
def test_watershed_segmentation(
    tmp_zarr, request, fixture_name, image_name, pseudo_neighborhood_radius_voxels
):
    # dynamically grab the correct fixture
    input_path = f"{tmp_zarr}/{image_name}/s0"
    output_path = f"{tmp_zarr}/test_watershed_segmentation_{image_name}"
    voxel_size = ImageDataInterface(input_path).ds.voxel_size[0]

    ws = FlawedWatershedSegmentation(
        input_path=input_path,
        output_path=output_path,
        pseudo_neighborhood_radius_nm=pseudo_neighborhood_radius_voxels * voxel_size,
        num_workers=1,
    )
    ws.get_watershed_segmentation()

    test_data = ImageDataInterface(f"{output_path}/s0").to_ndarray_ts()

    ground_truth_original_segmentation = request.getfixturevalue(fixture_name)
    fastremap.renumber(
        ground_truth_original_segmentation, in_place=True
    )  # so not uint64 if not necessary
    ground_truth = global_watershed(
        ground_truth_original_segmentation.astype(np.uint8),
        pseudo_neighborhood_radius_voxels,
        voxel_size,
    )
    assert arrays_equal_up_to_id_ordering(
        test_data,
        ground_truth,
    )

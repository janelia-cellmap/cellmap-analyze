from cellmap_analyze.process.mutex_watershed import MutexWatershed
import fastremap
import numpy as np

from cellmap_analyze.util.image_data_interface import (
    ImageDataInterface,
)


def test_connected_components(
    tmp_zarr,
    segmentation_cylinders,
):

    mw = MutexWatershed(
        affinities_path=f"{tmp_zarr}/affinities_cylinders",
        output_path=f"{tmp_zarr}/test_mws",
        adjacent_edge_bias=0,
        lr_bias_ratio=0,
        filter_val=0.05,
        connectivity=3,
        num_workers=1,
        delete_tmp=True,
    )
    mw.get_connected_components()
    test_data = ImageDataInterface(f"{tmp_zarr}/test_mws/s0").to_ndarray_ts()
    ground_truth = segmentation_cylinders.copy()
    ground_truth[0, 0, 0] = 0  # since we set it to nonzero in its generation
    test_uniques = fastremap.unique(test_data[test_data > 0])
    relabeling_dict = {}

    correct_uniques = True
    for test_unique in test_uniques:
        z, y, x = np.nonzero(test_data == test_unique)
        cooresponding_gt_uniques = ground_truth[z, y, x]
        if len(fastremap.unique(cooresponding_gt_uniques)) > 1:
            correct_uniques = False
            break
        relabeling_dict[test_unique] = ground_truth[z[0], y[0], x[0]]

    fastremap.remap(
        test_data, relabeling_dict, preserve_missing_labels=True, in_place=True
    )
    assert correct_uniques and np.array_equal(test_data, ground_truth)

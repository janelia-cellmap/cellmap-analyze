from cellmap_analyze.process.mutex_watershed import MutexWatershed
import fastremap
import numpy as np

from cellmap_analyze.util.image_data_interface import (
    ImageDataInterface,
)
from tests.test_utils import arrays_equal_up_to_id_ordering


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
    assert arrays_equal_up_to_id_ordering(test_data, ground_truth)

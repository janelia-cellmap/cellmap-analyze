import pytest
from cellmap_analyze.process.mutex_watershed import MutexWatershed
import fastmorph
import fastremap
from cellmap_analyze.util.xarray_image_data_interface import (
    XarrayImageDataInterface,
)
from tests.test_utils import arrays_equal_up_to_id_ordering


@pytest.mark.parametrize("do_opening", [True, False])
def test_mutex_watershed(tmp_zarr, segmentation_cylinders, do_opening):

    mw = MutexWatershed(
        affinities_path=f"{tmp_zarr}/affinities_cylinders",
        output_path=f"{tmp_zarr}/test_mws_{do_opening}",
        adjacent_edge_bias=0,
        lr_bias_ratio=0,
        filter_val=0.05,
        connectivity=3,
        num_workers=1,
        delete_tmp=True,
        do_opening=do_opening,
    )
    mw.get_connected_components()
    test_data = XarrayImageDataInterface(
        f"{tmp_zarr}/test_mws_{do_opening}/s0"
    ).to_ndarray_ts()
    ground_truth = segmentation_cylinders.copy()

    if do_opening:
        ground_truth = fastmorph.erode(ground_truth)
        ground_truth = fastmorph.dilate(ground_truth)
        fastremap.renumber(ground_truth, in_place=True)

    ground_truth[0, 0, 0] = 0  # since we set it to nonzero in its generation
    assert arrays_equal_up_to_id_ordering(test_data, ground_truth)

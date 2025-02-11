import numpy as np
from cellmap_analyze.process.filter_ids import FilterIDs
from cellmap_analyze.util.image_data_interface import ImageDataInterface


def test_filter_ids(tmp_zarr, image_with_holes_filled):
    ids_to_keep = [1, 4]
    fi = FilterIDs(
        input_path=f"{tmp_zarr}/image_with_holes_filled/s0",
        ids_to_keep=ids_to_keep,
        num_workers=1,
    )
    fi.filter_ids()
    ground_truth = np.zeros_like(image_with_holes_filled, dtype=np.uint8)
    for idx, id_to_keep in enumerate(ids_to_keep):
        ground_truth[image_with_holes_filled == id_to_keep] = idx + 1

    test_data = ImageDataInterface(
        f"{tmp_zarr}/image_with_holes_filled_filteredIDs/s0"
    ).to_ndarray_ts()

    assert np.array_equal(
        test_data,
        ground_truth,
    )

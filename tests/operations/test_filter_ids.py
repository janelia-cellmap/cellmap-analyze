import pytest
import numpy as np
from cellmap_analyze.process.filter_ids import FilterIDs
from cellmap_analyze.util.image_data_interface import ImageDataInterface
import pandas as pd


@pytest.mark.parametrize(
    "ids_to_keep, binarize",
    [
        ([1, 2], True),
        ([1, 2], False),
        ("tmp_object_information_csv", True),
        ("tmp_object_information_csv", False),
    ],
)
def test_filter_ids(ids_to_keep, binarize, tmp_zarr, image_with_holes_filled, request):
    if isinstance(ids_to_keep, str):
        ids_to_keep = request.getfixturevalue(ids_to_keep)  # Resolve fixture

    fi = FilterIDs(
        input_path=f"{tmp_zarr}/image_with_holes_filled/s0",
        ids_to_keep=ids_to_keep,
        num_workers=1,
        binarize=binarize,
    )
    fi.get_filtered_ids()
    if type(ids_to_keep) == str:
        ids_to_keep = pd.read_csv(ids_to_keep)["Object ID"].to_list()

    ground_truth = np.zeros_like(image_with_holes_filled, dtype=np.uint8)
    for idx, id_to_keep in enumerate(ids_to_keep):
        ground_truth[image_with_holes_filled == id_to_keep] = idx + 1

    if binarize:
        ground_truth = (ground_truth > 0).astype(np.uint8)

    test_data = ImageDataInterface(
        f"{tmp_zarr}/image_with_holes_filled_filteredIDs/s0"
    ).to_ndarray_ts()

    assert np.array_equal(
        test_data,
        ground_truth,
    )


def test_filter_ids_preserve_ids(
    tmp_zarr, image_with_holes_filled, tmp_object_information_csv
):
    # the csv keeps ids 1 and 4, so renumbering would have written 4 as 2
    fi = FilterIDs(
        input_path=f"{tmp_zarr}/image_with_holes_filled/s0",
        output_path=f"{tmp_zarr}/image_with_holes_filled_preservedIDs",
        ids_to_keep=tmp_object_information_csv,
        num_workers=1,
        preserve_ids=True,
    )
    fi.get_filtered_ids()

    ground_truth = np.where(
        np.isin(image_with_holes_filled, [1, 4]), image_with_holes_filled, 0
    )
    test_data = ImageDataInterface(
        f"{tmp_zarr}/image_with_holes_filled_preservedIDs/s0"
    ).to_ndarray_ts()

    assert np.any(test_data == 4)
    assert np.array_equal(test_data, ground_truth)


def test_filter_ids_binarize_and_preserve_ids_conflict(tmp_zarr):
    with pytest.raises(ValueError, match="mutually exclusive"):
        FilterIDs(
            input_path=f"{tmp_zarr}/image_with_holes_filled/s0",
            ids_to_keep=[1, 4],
            binarize=True,
            preserve_ids=True,
        )

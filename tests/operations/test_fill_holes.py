from cellmap_analyze.process.fill_holes import FillHoles

import numpy as np

from cellmap_analyze.util.image_data_interface import (
    ImageDataInterface,
)


def test_fill_holes(
    tmp_zarr,
    image_with_holes_filled,
):
    fh = FillHoles(
        input_path=f"{tmp_zarr}/image_with_holes/s0",
        output_path=f"{tmp_zarr}/test_fill_holes",
        num_workers=1,
        connectivity=1,
    )
    fh.fill_holes()

    ground_truth = image_with_holes_filled
    test_data = ImageDataInterface(f"{tmp_zarr}/test_fill_holes/s0").to_ndarray_ts()
    assert np.array_equal(
        test_data,
        ground_truth,
    )


# # %%
# import cellmap_analyze.process.fill_holes
# from importlib import reload

# reload(cellmap_analyze.process.fill_holes)
# from cellmap_analyze.process.fill_holes import FillHoles

# fh = FillHoles(
#     input_path=f"/tmp/pytest-of-ackermand/pytest-current/tmp0/tmp.zarr/image_with_holes/s0",
#     output_path=f"/tmp/pytest-of-ackermand/pytest-current/tmp0/tmp.zarr/test_fill_holes",
#     num_workers=1,
#     connectivity=1,
# )
# fh.fill_holes()

# # %%
# from cellmap_analyze.util.image_data_interface import ImageDataInterface
# from cellmap_analyze.util.visualization_util import view_in_neuroglancer

# view_in_neuroglancer(
#     image=ImageDataInterface(
#         f"/tmp/pytest-of-ackermand/pytest-current/tmp0/tmp.zarr/image_with_holes/s0"
#     ).to_ndarray_ts(),
#     holes=ImageDataInterface(
#         f"/tmp/pytest-of-ackermand/pytest-current/tmp0/tmp.zarr/test_fill_holes_holes/s0"
#     ).to_ndarray_ts(),
#     filled=ImageDataInterface(
#         f"/tmp/pytest-of-ackermand/pytest-current/tmp0/tmp.zarr/test_fill_holes/s0"
#     ).to_ndarray_ts(),
# )
# # %%

from cellmap_analyze.util.dask_util import delete_tmp_dir_blockwise
import os


def test_delete_tmp_zarr(
    tmp_zarr,
):
    zarr_path = f"{tmp_zarr}/random_image_to_delete/s0"
    delete_tmp_dir_blockwise(
        zarr_path,
        num_workers=1,
        compute_args={"scheduler": "single-threaded"},
    )
    assert os.path.exists(f"{tmp_zarr}/random_image_to_delete") is False

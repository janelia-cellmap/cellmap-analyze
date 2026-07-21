from cellmap_analyze.util.dask_util import (
    delete_tmp_dir_blockwise,
    get_zarr_chunk_path_from_block_index,
    get_num_blocks,
)
from cellmap_analyze.util.image_data_interface import ImageDataInterface
import os


def test_zarr_chunk_path_matches_real_chunk_files(tmp_zarr):
    # get_zarr_chunk_path_from_block_index must match the array's actual
    # on-disk chunk-key encoding (e.g. zarr v3's "c/" prefix); this runs
    # before test_delete_tmp_zarr so it checks real, undeleted chunk files.
    idi = ImageDataInterface(f"{tmp_zarr}/random_image_to_delete/s0")
    num_blocks = get_num_blocks(idi, idi.roi)
    assert num_blocks > 1

    found_real_chunk = any(
        os.path.exists(get_zarr_chunk_path_from_block_index(block_index, idi, depth=3))
        for block_index in range(num_blocks)
    )
    assert found_real_chunk, (
        "get_zarr_chunk_path_from_block_index produced no path matching a "
        "real on-disk chunk file - check it against the array's actual "
        "chunk-key encoding"
    )


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

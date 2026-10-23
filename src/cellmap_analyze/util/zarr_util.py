import os
import shutil

import worldarray as wa

from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.io_util import split_dataset_path


def create_multiscale_dataset(output_path, dtype, grid, scale=0):
    """Create level ``s{scale}`` of an OME-Zarr multiscale group at ``output_path``
    on ``grid`` (its voxels, placement and chunks), replacing the group when it
    exists. Returns the writable ``wa.WorldArray``."""
    filename, dataset = split_dataset_path(output_path, scale=scale)
    if ("zarr" in filename or "n5" in filename) and os.path.exists(output_path):
        shutil.rmtree(output_path)
    return wa.create(
        f"{filename}/{dataset}",
        grid.shape,
        dtype,
        grid.voxel_size,
        grid.translation,
        chunks=grid.chunks,
    )


def create_multiscale_dataset_idi(
    output_path,
    dtype,
    grid,
    scale=0,
    custom_fill_value=None,
    chunk_shape=None,
):
    create_multiscale_dataset(output_path, dtype, grid, scale=scale)
    return ImageDataInterface(
        output_path + f"/s{scale}",
        mode="r+",
        custom_fill_value=custom_fill_value,
        chunk_shape=chunk_shape,
    )

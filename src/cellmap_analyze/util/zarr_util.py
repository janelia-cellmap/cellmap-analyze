import os
import shutil

import numpy as np
import worldarray as wa
from funlib.geometry import Coordinate

from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.io_util import split_dataset_path
from cellmap_analyze.util.voxel_size_utils import scale_voxel_size_to_integers


def create_multiscale_dataset(
    output_path,
    dtype,
    voxel_size,
    total_roi,
    write_size,
    scale=0,
    original_voxel_size=None,
):
    """Create level ``s{scale}`` of an OME-Zarr multiscale group at ``output_path``,
    replacing the group when it exists. ``voxel_size``, ``total_roi`` and
    ``write_size`` (the chunk) are in the integer frame; ``original_voxel_size`` is
    the true voxel size that frame scales. Returns the writable ``wa.WorldArray``."""
    filename, dataset = split_dataset_path(output_path, scale=scale)
    if ("zarr" in filename or "n5" in filename) and os.path.exists(output_path):
        shutil.rmtree(output_path)

    voxel_size = Coordinate(voxel_size)
    if original_voxel_size is not None:
        true_voxel_size = np.array(original_voxel_size, dtype=float)
        # The caller's frame factor, which can exceed the voxel size's own
        # when the corner needs it (e.g. 5 nm voxels centred on 0).
        scale_factor = int(round(voxel_size[0] / original_voxel_size[0]))
    else:
        true_voxel_size = np.array(voxel_size, dtype=float)
        _, scale_factor = scale_voxel_size_to_integers(list(voxel_size))
    corner = np.array(total_roi.begin, dtype=float) / scale_factor
    return wa.create(
        f"{filename}/{dataset}",
        tuple(total_roi.shape / voxel_size),
        dtype,
        true_voxel_size.tolist(),
        # OME's translation is the centre of voxel 0
        (corner + true_voxel_size / 2).tolist(),
        chunks=tuple(Coordinate(write_size) / voxel_size),
    )


def create_multiscale_dataset_idi(
    output_path,
    dtype,
    voxel_size,
    total_roi,
    write_size,
    scale=0,
    custom_fill_value=None,
    chunk_shape=None,
    original_voxel_size=None,
):
    create_multiscale_dataset(
        output_path,
        dtype,
        voxel_size,
        total_roi,
        write_size,
        scale=scale,
        original_voxel_size=original_voxel_size,
    )
    return ImageDataInterface(
        output_path + f"/s{scale}",
        mode="r+",
        custom_fill_value=custom_fill_value,
        chunk_shape=chunk_shape,
    )

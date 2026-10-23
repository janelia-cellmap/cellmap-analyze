import os
import shutil

import worldarray as wa

from cellmap_analyze.util.io_util import split_dataset_path


def concurrency_context(concurrency_limit):
    """The per-handle context cellmap-analyze has always used: data copies and file IO
    capped, network request pools left at tensorstore's defaults."""
    if not concurrency_limit:
        return None
    return wa.capped_context(concurrency=concurrency_limit, network=False)


def open_dataset(
    path,
    mode="r",
    padding=0,
    method="nearest",
    concurrency_limit=1,
    max_retries=10,
    timeout=5,
    voxel_size=None,
):
    """The array at ``path`` as a ``wa.WorldArray``: for a multiscale group its first
    (finest) level, or with ``voxel_size`` the coarsest level no coarser than it (the
    first when every level is coarser), at that level's own voxel size. Reads outside
    the data are filled with ``padding`` (a value, or "edge")."""
    kwargs = dict(
        mode=mode,
        padding=padding,
        method=method,
        context=concurrency_context(concurrency_limit),
        retries=wa.Retry(max_attempts=max_retries, timeout=timeout),
    )
    if voxel_size is not None:
        return wa.open(
            str(path),
            voxel_size=tuple(float(v) for v in voxel_size),
            level_mode="floor",
            on_mismatch="keep",
            **kwargs,
        )
    return wa.open(str(path), level="first", **kwargs)


def create_multiscale_dataset(output_path, dtype, grid, scale=0, **open_kwargs):
    """Create level ``s{scale}`` of an OME-Zarr multiscale group at ``output_path``
    on ``grid`` (its voxels, placement and chunks), replacing the group when it
    exists. Returns it opened for writing (``open_kwargs`` as for ``open_dataset``)."""
    filename, dataset = split_dataset_path(output_path, scale=scale)
    if ("zarr" in filename or "n5" in filename) and os.path.exists(output_path):
        shutil.rmtree(output_path)
    wa.create(
        f"{filename}/{dataset}",
        grid.shape,
        dtype,
        grid.voxel_size,
        grid.translation,
        chunks=grid.chunks,
    )
    return open_dataset(f"{output_path}/s{scale}", mode="r+", **open_kwargs)

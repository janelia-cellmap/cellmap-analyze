import os
import shutil

import tensorstore as ts
import worldarray as wa

from cellmap_analyze.util.io_util import split_dataset_path


def concurrency_context(concurrency_limit=None):
    """The per-handle context cellmap-analyze uses: data copies and file IO capped at
    ``concurrency_limit``, network request pools left at tensorstore's defaults. Without
    a limit both pools take ``wa.thread_budget()`` of the process the handle is used in,
    so a handle opened on the driver is sized again in each worker: its share of the
    job's CPUs (``dask_util.worker_thread_budget``), else every CPU this process may use.
    worldarray's contexts write chunks as zarr-python made them, without a lock or an
    fsync; every writer owns whole chunks no other writer touches."""
    if concurrency_limit:
        return ts.Context(
            wa.capped_context_spec(concurrency=concurrency_limit, network=False)
        )
    return wa.shared_context(0, file_io_floor=1)


def open_dataset(
    path,
    mode="r",
    padding=0,
    method="nearest",
    concurrency_limit=None,
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
        context=concurrency_context(None),
    )
    return open_dataset(f"{output_path}/s{scale}", mode="r+", **open_kwargs)

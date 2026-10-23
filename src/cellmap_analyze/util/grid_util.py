"""Grids: where an operation's voxels sit in the world (nm, zyx).

Blockwise code works in the voxel indices of one grid per operation, and every dataset
it reads is read onto that grid (``WorldArray.read_onto``), so no two datasets need to
share a voxel size or a corner."""

import numpy as np
import worldarray as wa


def _snap(values, tol=1e-6):
    """``values`` with those within ``tol`` of a whole number rounded to it."""
    values = np.asarray(values, dtype=float)
    rounded = np.round(values)
    return np.where(np.abs(values - rounded) <= tol, rounded, values)


def make_grid(shape, voxel_size, corner=None, chunks=None):
    """``shape`` voxels of ``voxel_size`` whose voxel 0 starts at ``corner`` (the
    origin by default), chunked by ``chunks``."""
    voxel_size = tuple(float(v) for v in voxel_size)
    if corner is None:
        corner = (0.0,) * len(voxel_size)
    return wa.Grid(
        tuple(int(s) for s in shape),
        voxel_size,
        wa.corner_to_centre(corner, voxel_size),
        chunks=None if chunks is None else tuple(int(c) for c in chunks),
    )


def grid_of(vol, chunks=None):
    """The spatial grid of ``vol`` (a ``wa.WorldArray``), chunked by ``chunks`` (its
    own chunks by default)."""
    spatial = vol.spatial
    if chunks is None:
        chunks = [vol.meta.chunk_shape[i] for i in spatial]
    return make_grid([vol.shape[i] for i in spatial], vol.voxel_size, vol.corner, chunks)


def voxel_corner(grid, index):
    """Where the corner of voxel ``index`` of ``grid`` sits."""
    return tuple(c + i * v for c, i, v in zip(grid.corner, index, grid.voxel_size))


def crop(grid, begin, end):
    """The voxels ``[begin, end)`` of ``grid`` as a grid of their own, chunked alike."""
    return make_grid(
        np.subtract(end, begin), grid.voxel_size, voxel_corner(grid, begin), grid.chunks
    )


def crop_to_roi(grid, roi):
    """The voxels of ``grid`` whose centres lie in ``roi``, a world box
    ``(begin, end)``; ``grid`` itself when ``roi`` is None."""
    if roi is None:
        return grid
    begin, end = grid.index_box(roi)
    begin = np.clip(begin, 0, grid.shape)
    end = np.clip(end, begin, grid.shape)
    return crop(grid, begin, end)


def intersection(grids, voxel_size, chunks=None):
    """A grid of ``voxel_size`` voxels over the world box every grid in ``grids``
    covers, from its corner; the voxels that do not fit whole are dropped."""
    box = grids[0].box
    for g in grids[1:]:
        box = box.intersect(g.box)
    shape = np.floor(_snap(np.asarray(box.shape) / np.asarray(voxel_size, dtype=float)))
    return make_grid(np.maximum(shape, 0), voxel_size, box.begin, chunks)

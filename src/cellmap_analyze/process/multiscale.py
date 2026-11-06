import logging
import os
import re

import dask.bag as db
import numpy as np
import tensorstore as ts
import worldarray as wa
import zarr

from cellmap_analyze.util import dask_util
from cellmap_analyze.util.io_util import TimingMessager, is_remote_path
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.zarr_util import concurrency_context

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Methods that reduce a whole window, so a level-k voxel is centred on the
# window of finer voxels it summarises (wa.level_placement): tensorstore's
# downsample methods, and those below that suppress zero.
_BLOCK_METHODS = ("mode", "mean", "median", "min", "max", "mode_suppress_zero", "mean_suppress_zero")
_DRIVERS = {"zarr2": "zarr", "zarr3": "zarr3"}


def _windows(data, factors):
    """The windows of ``data`` that each level voxel summarises, one per row,
    and the level's shape. A window past the edge of ``data`` is padded with
    zeros, which the methods below ignore."""
    factors = np.array(factors)
    shape = -(-np.array(data.shape) // factors)
    padded = np.zeros(tuple(shape * factors), data.dtype)
    padded[tuple(slice(0, s) for s in data.shape)] = data
    windows = padded.reshape(
        shape[0], factors[0], shape[1], factors[1], shape[2], factors[2]
    ).transpose(0, 2, 4, 1, 3, 5).reshape(-1, int(factors.prod()))
    return windows, tuple(shape)


def mode_suppress_zero(data, factors):
    """``data`` downsampled by ``factors``, each window taking its most common
    non-zero label (the smallest on a tie) and 0 only if it has none, as
    neuclease's ``flat_mode_except_zero`` does, so thin objects such as
    skeletons and medial surfaces survive."""
    windows, shape = _windows(data, factors)
    windows = np.sort(windows, axis=1)
    counts = np.stack(
        [(windows == windows[:, [j]]).sum(axis=1) for j in range(windows.shape[1])], axis=1
    )
    counts[windows == 0] = 0
    return windows[np.arange(len(windows)), counts.argmax(axis=1)].reshape(shape)


def mean_suppress_zero(data, factors):
    """``data`` downsampled by ``factors``, each window taking the mean of its
    non-zero voxels (rounded for integers) and 0 only if it has none, so
    values held only on thin objects are neither lost nor diluted."""
    windows, shape = _windows(data, factors)
    counts = np.count_nonzero(windows, axis=1)
    mean = windows.sum(axis=1, dtype=np.float64) / np.maximum(counts, 1)
    if np.issubdtype(data.dtype, np.integer):
        mean = np.rint(mean)
    return mean.astype(data.dtype).reshape(shape)


_SUPPRESS_ZERO = {"mode_suppress_zero": mode_suppress_zero, "mean_suppress_zero": mean_suppress_zero}


def _level_index(path):
    match = re.fullmatch(r"s(\d+)", path)
    return int(match.group(1)) if match else None


class Multiscale(ComputeConfigMixin):
    """Add downsampled levels (s1, s2, ...) to an existing OME-Zarr dataset.

    Each level is computed blockwise from the previous one with tensorstore's
    ``downsample`` driver: by default ``mode`` for integer data, which keeps
    label ids intact, and ``mean`` for floats. ``mode_suppress_zero`` and
    ``mean_suppress_zero`` leave zeros out of each window, so thin objects
    that ``mode`` would lose to the background, and the values on them, survive. Levels are written until one
    fits in a single chunk, or ``num_levels`` of them. An axis is only halved
    once its voxel size is within 1.5x of the finest (``wa.level_factors``),
    so anisotropic data moves toward isotropic first.
    """

    def __init__(
        self,
        input_path,
        num_levels=None,
        downsample_method=None,
        overwrite=False,
        num_workers=10,
    ):
        super().__init__(num_workers)
        if downsample_method is not None and downsample_method not in _BLOCK_METHODS:
            raise ValueError(
                f"downsample_method must be one of {_BLOCK_METHODS}, got {downsample_method!r}"
            )

        # Levels are written beside the input, so it must be local; made
        # absolute so dask workers, whose working directory may differ, open
        # the same files.
        input_path = str(input_path).rstrip("/")
        if is_remote_path(input_path):
            raise ValueError(f"multiscale writes next to its input, which must be local: {input_path}")
        input_path = os.path.abspath(input_path)
        # Accept the multiscale group (base level s0) or one of its sN levels.
        group_path, leaf = os.path.split(input_path)
        if _level_index(leaf) is None:
            group_path, leaf = input_path, "s0"
        self.group_path = group_path
        self.base_level = _level_index(leaf)

        self.base = wa.open(f"{self.group_path}/{leaf}")
        if self.base.meta.format not in _DRIVERS:
            raise ValueError(
                f"multiscale needs a zarr v2 or v3 dataset, got {self.base.meta.format} at {input_path}"
            )
        if self.base.meta.ndim != 3:
            raise ValueError(f"multiscale supports 3D arrays, got shape {self.base.shape}")
        self.chunks = tuple(self.base.meta.chunk_shape)

        if downsample_method is None:
            floating = np.issubdtype(self.base.dtype, np.floating)
            downsample_method = "mean" if floating else "mode"
        self.downsample_method = downsample_method

        self.levels = self._plan_levels(num_levels)
        existing = [
            level["name"]
            for level in self.levels
            if os.path.exists(f"{self.group_path}/{level['name']}")
        ]
        if existing and not overwrite:
            raise FileExistsError(
                f"{self.group_path} already has {existing}; pass overwrite=True to rebuild them"
            )

    def _plan_levels(self, num_levels):
        shape = self.base.shape
        voxel_size, translation = self.base.voxel_size, self.base.translation
        levels = []
        for factors in wa.level_factors(
            voxel_size,
            max_levels=num_levels,
            stop_shape=self.chunks if num_levels is None else None,
            shape=shape,
        ):
            shape = wa.level_shape(shape, factors)
            voxel_size, translation = wa.level_placement(voxel_size, translation, factors)
            levels.append(
                {
                    "name": f"s{self.base_level + len(levels) + 1}",
                    "factors": tuple(factors),
                    "shape": tuple(shape),
                    "voxel_size": tuple(voxel_size),
                    "translation": tuple(translation),
                }
            )
        return levels

    @staticmethod
    def downsample_block(
        block_index, source_path, dest_path, driver, factors, blocks, method
    ):
        block = blocks[block_index]
        context = concurrency_context()
        source = wa.open(source_path, context=context)
        # The finer voxels under this block, clipped to the array: an edge
        # window past it is summarised from the voxels it has.
        begin = [b * f for b, f in zip(block.begin, factors)]
        end = [min(e * f, n) for e, f, n in zip(block.end, factors, source.shape)]
        data = source.read_index(begin, end, retries=wa.Retry())
        if method in _SUPPRESS_ZERO:
            data = _SUPPRESS_ZERO[method](data, factors)
        else:
            data = ts.downsample(ts.array(data), list(factors), method).read().result()
        # Plain tensorstore: the level is not listed in the group until it is complete.
        dest = ts.open(
            {"driver": driver, "kvstore": {"driver": "file", "path": dest_path}},
            write=True,
            context=context,
        ).result()
        dest[tuple(slice(b, e) for b, e in zip(block.begin, block.end))].write(data).result()

    def _create_level_array(self, level):
        zarr.open_array(
            store=f"{self.group_path}/{level['name']}",
            shape=level["shape"],
            chunks=self.chunks,
            dtype=self.base.dtype,
            fill_value=self.base.meta.fill_value,
            mode="w",
            zarr_format=2 if self.base.meta.format == "zarr2" else 3,
        )

    def _list_level(self, name, voxel_size, translation):
        wa.ensure_group_entry(self.group_path, name, voxel_size, translation, self.base.axes)

    def create_multiscale(self):
        source_path = f"{self.group_path}/s{self.base_level}"
        if not self.levels:
            logger.info(f"{source_path} already fits in one chunk; no levels to add")
            return
        # A base placed only by its own attributes gets its entry first.
        self._list_level(f"s{self.base_level}", self.base.voxel_size, self.base.translation)
        # One cluster for every level; each reads the one before, so they run in turn.
        with dask_util.start_dask(self.num_workers, f"multiscale of {source_path}", logger):
            for level in self.levels:
                dest_path = f"{self.group_path}/{level['name']}"
                self._create_level_array(level)
                blocks = wa.blocks(level["shape"], self.chunks)
                bag = db.range(
                    len(blocks),
                    npartitions=dask_util.guesstimate_npartitions(len(blocks), self.num_workers),
                ).map_partitions(
                    dask_util.partition_worker,
                    Multiscale.downsample_block,
                    source_path,
                    dest_path,
                    _DRIVERS[self.base.meta.format],
                    level["factors"],
                    blocks,
                    self.downsample_method,
                )
                with TimingMessager(f"Downsampling {source_path} to {level['name']}", logger):
                    bag.compute(**self.compute_args)
                # List each level only once its data is complete.
                self._list_level(level["name"], level["voxel_size"], level["translation"])
                source_path = dest_path

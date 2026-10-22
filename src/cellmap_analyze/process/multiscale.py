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

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# tensorstore downsample methods that reduce a whole window, so a level-k voxel
# is centred on the window of finer voxels it summarises (wa.level_placement).
_BLOCK_METHODS = ("mode", "mean", "median", "min", "max")
_DRIVERS = {"zarr2": "zarr", "zarr3": "zarr3"}


def _level_index(path):
    match = re.fullmatch(r"s(\d+)", path)
    return int(match.group(1)) if match else None


class Multiscale(ComputeConfigMixin):
    """Add downsampled levels (s1, s2, ...) to an existing OME-Zarr dataset.

    Each level is computed blockwise from the previous one with tensorstore's
    ``downsample`` driver: by default ``mode`` for integer data, which keeps
    label ids intact, and ``mean`` for floats. Levels are written until one
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
        self.concurrency_limit = dask_util.resolve_concurrency_limit(num_workers)

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
        block_index, source_path, dest_path, driver, factors, blocks, method, concurrency_limit
    ):
        block = blocks[block_index]
        context = wa.capped_context(concurrency=concurrency_limit, network=False)
        source = wa.open(source_path, context=context)
        # The finer voxels under this block, clipped to the array: an edge
        # window past it is summarised from the voxels it has.
        begin = [b * f for b, f in zip(block.begin, factors)]
        end = [min(e * f, n) for e, f, n in zip(block.end, factors, source.shape)]
        data = source.read_index(begin, end, retries=wa.Retry())
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
                    self.concurrency_limit,
                )
                with TimingMessager(f"Downsampling {source_path} to {level['name']}", logger):
                    bag.compute(**self.compute_args)
                # List each level only once its data is complete.
                self._list_level(level["name"], level["voxel_size"], level["translation"])
                source_path = dest_path

import logging
import os
import re

import numpy as np
import tensorstore as ts
import zarr

from cellmap_analyze.util import dask_util
from cellmap_analyze.util.image_data_interface import (
    ImageDataInterface,
    open_ds_tensorstore,
    read_with_retries,
)
from cellmap_analyze.util.io_util import is_remote_path
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.voxel_size_utils import (
    get_multiscales,
    read_raw_offset,
    read_raw_voxel_size,
)
from cellmap_analyze.util.zarr_util import generate_multiscales_metadata

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# tensorstore downsample methods that reduce a whole window, so a level-k voxel
# is centred on the window of finer voxels it summarises (see _plan_levels).
_BLOCK_METHODS = ("mode", "mean", "median", "min", "max")


def downsample_factors(voxel_size, anisotropy_tolerance=1.5):
    """Per-axis factor from one level to the next.

    2 along every axis whose voxel size is within ``anisotropy_tolerance`` of
    the finest, else 1, so anisotropic data moves toward isotropic before all
    axes halve together (mesh-n-bone's rule, so the two pyramids agree).
    """
    voxel_size = np.asarray(voxel_size, dtype=float)
    return tuple(
        2 if vs <= anisotropy_tolerance * voxel_size.min() else 1
        for vs in voxel_size
    )


def _level_index(path):
    match = re.fullmatch(r"s(\d+)", path)
    return int(match.group(1)) if match else None


class Multiscale(ComputeConfigMixin):
    """Add downsampled levels (s1, s2, ...) to an existing OME-Zarr dataset.

    Each level is computed blockwise from the previous one with tensorstore's
    ``downsample`` driver; the default ``mode`` keeps label ids intact. Levels
    are written until one fits in a single chunk, or ``num_levels`` of them.
    """

    def __init__(
        self,
        input_path,
        num_levels=None,
        downsample_method="mode",
        overwrite=False,
        num_workers=10,
    ):
        super().__init__(num_workers)
        if downsample_method not in _BLOCK_METHODS:
            raise ValueError(
                f"downsample_method must be one of {_BLOCK_METHODS}, got {downsample_method!r}"
            )
        self.downsample_method = downsample_method

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

        base_idi = ImageDataInterface(f"{self.group_path}/{leaf}")
        if base_idi.filetype not in ("zarr", "zarr3"):
            raise ValueError(
                f"multiscale needs a zarr v2 or v3 dataset, got {base_idi.filetype} at {input_path}"
            )
        base_array = base_idi.ds.data
        if base_array.ndim != 3:
            raise ValueError(f"multiscale supports 3D arrays, got shape {base_array.shape}")
        self.chunks = tuple(base_array.chunks)
        self.dtype = base_array.dtype
        self.fill_value = base_array.fill_value
        # True nm voxel size and OME translation (voxel 0's centre) of the base.
        self.voxel_size = read_raw_voxel_size(base_idi.ds)
        self.translation = read_raw_offset(base_idi.ds)

        self.levels = self._plan_levels(base_array.shape, num_levels)
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

    def _plan_levels(self, shape, num_levels):
        shape = np.array(shape)
        voxel_size = np.array(self.voxel_size, dtype=float)
        translation = np.array(self.translation, dtype=float)
        levels = []
        while True:
            if num_levels is not None and len(levels) >= num_levels:
                break
            if num_levels is None and np.all(shape <= np.array(self.chunks)):
                break
            factors = np.array(downsample_factors(voxel_size))
            shape = -(-shape // factors)
            # A block method's voxel summarises `factors` finer voxels, so its
            # centre is the centre of that window: half a finer voxel further
            # in per extra voxel. Summed over levels this is
            # translation_0 + voxel_size_0 * (F - 1) / 2 for a total factor F.
            translation = translation + voxel_size * (factors - 1) / 2
            voxel_size = voxel_size * factors
            levels.append(
                {
                    "name": f"s{self.base_level + len(levels) + 1}",
                    "factors": tuple(int(f) for f in factors),
                    "shape": tuple(int(s) for s in shape),
                    "voxel_size": voxel_size.tolist(),
                    "translation": translation.tolist(),
                }
            )
        return levels

    @staticmethod
    def downsample_block(
        block_index, source_path, dest_path, factors, chunks, dest_shape, method, concurrency_limit
    ):
        grid = [-(-s // c) for s, c in zip(dest_shape, chunks)]
        coords = np.unravel_index(block_index, grid)
        slices = tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, c, s in zip(coords, chunks, dest_shape)
        )
        source = ts.downsample(
            open_ds_tensorstore(source_path, concurrency_limit=concurrency_limit),
            list(factors),
            method,
        )
        data = read_with_retries(source, slices)
        dest = open_ds_tensorstore(dest_path, mode="r+", concurrency_limit=concurrency_limit)
        dest[slices].write(data).result()

    def _create_level_array(self, level):
        group = zarr.open_group(self.group_path, mode="r+")
        array = zarr.open_array(
            store=f"{self.group_path}/{level['name']}",
            shape=level["shape"],
            chunks=self.chunks,
            dtype=self.dtype,
            fill_value=self.fill_value,
            mode="w",
            zarr_format=group.metadata.zarr_format,
        )
        # Same per-array convention as create_multiscale_dataset.
        array.attrs["voxel_size"] = level["voxel_size"]
        array.attrs["offset"] = level["translation"]

    def _write_level_metadata(self, level):
        group = zarr.open_group(self.group_path, mode="r+")
        attrs = group.attrs.asdict()
        if get_multiscales(attrs) is None:
            # Base level described only by per-array attrs: start an OME entry for it.
            attrs.update(
                generate_multiscales_metadata(
                    f"s{self.base_level}",
                    list(self.voxel_size),
                    list(self.translation),
                    "nanometer",
                    ["z", "y", "x"],
                )
            )
        multiscale = get_multiscales(attrs)[0]
        datasets = [d for d in multiscale["datasets"] if d.get("path") != level["name"]]
        datasets.append(
            {
                "coordinateTransformations": [
                    {"scale": level["voxel_size"], "type": "scale"},
                    {"translation": level["translation"], "type": "translation"},
                ],
                "path": level["name"],
            }
        )
        datasets.sort(
            key=lambda d: (_level_index(d["path"]) is None, _level_index(d["path"]) or 0)
        )
        multiscale["datasets"] = datasets
        group.attrs.update(attrs)

    def create_multiscale(self):
        source_path = f"{self.group_path}/s{self.base_level}"
        if not self.levels:
            logger.info(f"{source_path} already fits in one chunk; no levels to add")
        for level in self.levels:
            dest_path = f"{self.group_path}/{level['name']}"
            self._create_level_array(level)
            num_blocks = int(
                np.prod([-(-s // c) for s, c in zip(level["shape"], self.chunks)])
            )
            dask_util.compute_blockwise_partitions(
                num_blocks,
                self.num_workers,
                self.compute_args,
                logger,
                f"downsampling {source_path} to {level['name']}",
                Multiscale.downsample_block,
                source_path,
                dest_path,
                level["factors"],
                self.chunks,
                level["shape"],
                self.downsample_method,
                self.concurrency_limit,
            )
            # List each level only once its data is complete.
            self._write_level_metadata(level)
            source_path = dest_path

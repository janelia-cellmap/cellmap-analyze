# %%
import numpy as np
from scipy import ndimage
from cellmap_analyze.util import dask_util
from cellmap_analyze.util.block_util import erosion
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
)
from cellmap_analyze.util.grid_util import crop_to_roi, grid_of

import logging
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.zarr_util import create_multiscale_dataset, open_dataset

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class LabelWithMask(ComputeConfigMixin):
    def __init__(
        self,
        input_path,
        mask_path,
        output_path,
        intensity_threshold_minimum=-1,
        intensity_threshold_maximum=np.inf,  # exclusive
        num_workers=10,
        surface_voxels_only=False,
        roi=None,
        chunk_shape=None,
    ):
        super().__init__(num_workers)

        self.input_vol = open_dataset(input_path)
        self.grid = crop_to_roi(grid_of(self.input_vol, chunk_shape), roi)
        # read on the input's grid
        self.mask_vol = open_dataset(
            mask_path,
            padding="edge",  # in case of doing erosion later for surface voxel testing
        )

        self.intensity_threshold_minimum = intensity_threshold_minimum
        self.intensity_threshold_maximum = intensity_threshold_maximum
        self.surface_voxels_only = surface_voxels_only
        self.output_vol = create_multiscale_dataset(
            output_path, dtype=self.mask_vol.dtype, grid=self.grid
        )

    @staticmethod
    def label_with_mask_blockwise(
        block_index,
        grid,
        input_vol,
        mask_vol,
        output_vol,
        intensity_threshold_minimum,
        intensity_threshold_maximum,
        surface_voxels_only=False,
    ):
        # the erosion below reaches one voxel on every axis
        block = create_block_from_index(
            grid,
            block_index,
            padding=int(surface_voxels_only),
        )
        input = input_vol.read_onto(grid, block.read_begin, block.read_end)
        mask = mask_vol.read_onto(grid, block.read_begin, block.read_end)
        if surface_voxels_only:
            eroded = erosion(mask, 1, ndimage.generate_binary_structure(3, 2))
            mask = mask - eroded
        output = (
            (input >= intensity_threshold_minimum)
            & (input < intensity_threshold_maximum)
        ) * mask
        output_vol.write_index(
            block.begin, block.end, output[block.inner].astype(output_vol.dtype, copy=False)
        )

    def get_label_with_mask(self):
        num_blocks = dask_util.get_num_blocks(self.grid)
        dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            f"labeling {self.input_vol.level_path} with mask {self.mask_vol.level_path}",
            LabelWithMask.label_with_mask_blockwise,
            self.grid,
            self.input_vol,
            self.mask_vol,
            self.output_vol,
            self.intensity_threshold_minimum,
            self.intensity_threshold_maximum,
            self.surface_voxels_only,
        )

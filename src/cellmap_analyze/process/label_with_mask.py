# %%
import numpy as np
from scipy import ndimage
from cellmap_analyze.util import dask_util
from cellmap_analyze.util.block_util import erosion
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
)
from cellmap_analyze.util.grid_util import crop_to_roi
from cellmap_analyze.util.image_data_interface import ImageDataInterface

import logging
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.zarr_util import create_multiscale_dataset_idi

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

        self.input_idi = ImageDataInterface(input_path, chunk_shape=chunk_shape)
        self.grid = crop_to_roi(self.input_idi.grid, roi)
        self.input_idi.output_grid = self.grid
        # read on the input's grid
        self.mask_idi = ImageDataInterface(
            mask_path,
            chunk_shape=chunk_shape,
            custom_fill_value="edge",  # in case of doing erosion later for surface voxel testing
            output_grid=self.grid,
        )

        self.intensity_threshold_minimum = intensity_threshold_minimum
        self.intensity_threshold_maximum = intensity_threshold_maximum
        self.surface_voxels_only = surface_voxels_only
        self.output_idi = create_multiscale_dataset_idi(
            output_path, dtype=self.mask_idi.dtype, grid=self.grid
        )

    @staticmethod
    def label_with_mask_blockwise(
        block_index,
        input_idi: ImageDataInterface,
        mask_idi: ImageDataInterface,
        output_idi: ImageDataInterface,
        intensity_threshold_minimum,
        intensity_threshold_maximum,
        surface_voxels_only=False,
    ):
        # the erosion below reaches one voxel on every axis
        block = create_block_from_index(
            output_idi,
            block_index,
            padding=int(surface_voxels_only),
        )
        input = input_idi.to_ndarray_ts(block.read_begin, block.read_end)
        mask = mask_idi.to_ndarray_ts(block.read_begin, block.read_end)
        if surface_voxels_only:
            eroded = erosion(mask, 1, ndimage.generate_binary_structure(3, 2))
            mask = mask - eroded
        output = (
            (input >= intensity_threshold_minimum)
            & (input < intensity_threshold_maximum)
        ) * mask
        output_idi.write(block.begin, block.end, output[block.inner])

    def get_label_with_mask(self):
        num_blocks = dask_util.get_num_blocks(self.output_idi)
        dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            f"labeling {self.input_idi.path} with mask {self.mask_idi.path}",
            LabelWithMask.label_with_mask_blockwise,
            self.input_idi,
            self.mask_idi,
            self.output_idi,
            self.intensity_threshold_minimum,
            self.intensity_threshold_maximum,
            self.surface_voxels_only,
        )

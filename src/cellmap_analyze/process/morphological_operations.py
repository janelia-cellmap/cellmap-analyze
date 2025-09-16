# %%
import numpy as np
from scipy import ndimage
from cellmap_analyze.util import dask_util
from cellmap_analyze.util.block_util import erosion
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
)
from cellmap_analyze.util.mask_util import MasksFromConfig
from cellmap_analyze.util.measure_util import trim_array
from cellmap_analyze.util.image_data_interface import ImageDataInterface

import logging
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.zarr_util import create_multiscale_dataset_idi

import fastmorph
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MorphologicalOperations(ComputeConfigMixin):
    def __init__(
        self,
        input_path,
        output_path,
        mask_config=None,
        num_workers=10,
        roi=None,
        chunk_shape=None,
        operation="erosion",
        iterations=1,
        connectivity=2,
    ):
        # NOTE: For dilation especially, it is not clear if simply padding by iterations is sufficient. May need to pad by more depending on structure sizes 
        # Notice that in tests, larger dilations wont be consistent though close. This may not be a big deal if only care about approximate morphology and exact consistency.
        super().__init__(num_workers)

        self.input_idi = ImageDataInterface(input_path, chunk_shape=chunk_shape)

        self.roi = roi
        if self.roi is None:
            self.roi = self.input_idi.roi
        
        self.mask = None
        if mask_config:
            self.mask = MasksFromConfig(
                mask_config,
                output_voxel_size=self.input_idi.voxel_size,
                connectivity=connectivity,
            )

        self.operation=operation
        if iterations<1:
            raise ValueError("iterations must be at least 1")
        
        self.iterations=iterations
        self.output_idi = create_multiscale_dataset_idi(
            output_path,
            dtype=self.input_idi.dtype,
            voxel_size=self.input_idi.voxel_size,
            total_roi=self.roi,
            write_size=self.input_idi.chunk_shape * self.input_idi.voxel_size,
        )

    @staticmethod
    def perform_morphological_operation_blockwise(
        block_index,
        input_idi: ImageDataInterface,
        output_idi: ImageDataInterface,
        operation: str,
        iterations: int,
        mask: MasksFromConfig = None,
    ):
        padding_voxels = iterations
        block = create_block_from_index(
            input_idi,
            block_index,
            padding=padding_voxels * input_idi.voxel_size[0],
        )
        if mask:
            mask_block = mask.process_block(roi=block.read_roi)
            if not np.any(mask_block):
                output_idi.ds[block.write_roi] = 0
                return
            
        data = input_idi.to_ndarray_ts(block.read_roi)
        if mask:
            data *= mask_block

        if operation=="erosion":
            data = fastmorph.erode(data, iterations=iterations)
        elif operation=="dilation":
            data = fastmorph.dilate(data, iterations=iterations)
        
        if mask:
            # need before and after to make sure nothing from outside mask makes it in and vice versa
            data *= mask_block
        output_idi.ds[block.write_roi] = trim_array(data, padding_voxels)

    def perform_morphological_operation(self):
        num_blocks = dask_util.get_num_blocks(self.input_idi, roi=self.roi)
        dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            f"{self.operation} of {self.input_idi.path}",
            MorphologicalOperations.perform_morphological_operation_blockwise,
            self.input_idi,
            self.output_idi,
            self.operation,
            self.iterations,
            self.mask,
        )

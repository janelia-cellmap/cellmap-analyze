import numpy as np
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
    dask_computer,
    guesstimate_npartitions,
)
from cellmap_analyze.util.image_data_interface import ImageDataInterface

import logging
import dask.bag as db
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
        roi=None,
        chunk_shape=None,
    ):
        super().__init__(num_workers)

        self.input_idi = ImageDataInterface(input_path, chunk_shape=chunk_shape)
        self.mask_idi = ImageDataInterface(mask_path, chunk_shape=chunk_shape)

        self.roi = roi
        if self.roi is None:
            self.roi = self.input_idi.roi

        self.intensity_threshold_minimum = intensity_threshold_minimum
        self.intensity_threshold_maximum = intensity_threshold_maximum
        self.output_idi = create_multiscale_dataset_idi(
            output_path,
            dtype=self.mask_idi.dtype,
            voxel_size=self.mask_idi.voxel_size,
            total_roi=self.roi,
            write_size=self.mask_idi.chunk_shape * self.mask_idi.voxel_size,
        )

    @staticmethod
    def label_with_mask_blockwise(
        block_index,
        input_idi: ImageDataInterface,
        mask_idi: ImageDataInterface,
        output_idi: ImageDataInterface,
        intensity_threshold_minimum,
        intensity_threshold_maximum,
    ):
        block = create_block_from_index(
            input_idi,
            block_index,
        )
        input = input_idi.to_ndarray_ts(block.read_roi)
        mask = mask_idi.to_ndarray_ts(block.read_roi)
        output = (
            (input >= intensity_threshold_minimum)
            & (input < intensity_threshold_maximum)
        ) * mask
        output_idi.ds[block.write_roi] = output

    def get_label_with_mask(self):
        num_blocks = dask_util.get_num_blocks(self.input_idi, roi=self.roi)
        block_indexes = list(range(num_blocks))
        b = db.from_sequence(
            block_indexes,
            npartitions=guesstimate_npartitions(block_indexes, self.num_workers),
        ).map(
            LabelWithMask.label_with_mask_blockwise,
            self.input_idi,
            self.mask_idi,
            self.output_idi,
            self.intensity_threshold_minimum,
            self.intensity_threshold_maximum,
        )

        with dask_util.start_dask(
            self.num_workers,
            "labeling with mask",
            logger,
        ):
            with io_util.Timing_Messager("Labeling with mask", logger):
                dask_computer(b, self.num_workers, **self.compute_args)

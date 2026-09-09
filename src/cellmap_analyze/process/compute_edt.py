import logging
import os

import edt as edt_module
import numpy as np
from funlib.geometry import Coordinate

from cellmap_analyze.util import dask_util
from cellmap_analyze.util.dask_util import create_block_from_index
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.io_util import split_dataset_path
from cellmap_analyze.util.mask_util import MasksFromConfig
from cellmap_analyze.util.measure_util import trim_array_anisotropic
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.zarr_util import create_multiscale_dataset_idi

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ComputeEDT(ComputeConfigMixin):
    """
    Blockwise, multi-label-aware Euclidean distance transform (EDT) over an
    entire labeled segmentation, persisted as its own dataset.

    ``edt.edt()`` (the Seung-lab package) is multi-label aware: given the raw
    labeled array directly -- not a binarized single-instance mask -- it
    computes, per voxel, the distance to the nearest voxel with a *different*
    label (background included). That is exactly the quantity
    ``SplitNarrowBridges`` computes per-object via ``data == id_value``, just
    for every instance in the dataset at once, in one blockwise pass.

    ``padding_nm`` is what bounds correctness, and it does so cheaply: a
    windowed EDT can only *overestimate* the true distance (it's a min over a
    smaller candidate set -- background outside the window is invisible to
    it), so a computed value is only ever too thick, never falsely thin. That
    means padding only needs to comfortably exceed whatever thinness
    threshold downstream consumers actually care about (e.g.
    ``SplitNarrowBridges``'s ``neck_radius_nm``), not the physical scale of
    any particular object -- large objects just get numerically inflated
    (too-large) distance values deep in their thick interiors, which is
    harmless for threshold-based ("is this thin?") uses. This is *not*
    sufficient for uses that need exact distance values everywhere (e.g. true
    peak-radius reporting on very thick objects); that would need much larger
    (or adaptively-verified, expand-until-provably-converged) padding.

    Bigger blocks amortize the fixed padding overhead better (measured: at a
    fixed 1000nm padding, doubling block size from 64 to 128 voxels/axis cut
    serial wall-time by ~3x on a 97M-voxel test object) -- ``block_multiplier``
    defaults to processing several native chunks per block for this reason.
    """

    def __init__(
        self,
        segmentation_path,
        output_path,
        padding_nm,
        block_shape_voxels=None,
        block_multiplier=4,
        mask_config=None,
        roi=None,
        num_workers=10,
        chunk_shape=None,
        delete_tmp=False,
    ):
        super().__init__(num_workers)
        self.segmentation_path = segmentation_path
        self.segmentation_idi = ImageDataInterface(
            segmentation_path, chunk_shape=chunk_shape
        )
        self.output_path = str(output_path).rstrip("/")
        os.makedirs(split_dataset_path(self.output_path)[0], exist_ok=True)
        self.roi = roi if roi is not None else self.segmentation_idi.roi

        self.padding_nm = float(padding_nm)

        if block_shape_voxels is not None:
            self.block_size = (
                Coordinate(block_shape_voxels) * self.segmentation_idi.voxel_size
            )
        else:
            self.block_size = (
                self.segmentation_idi.chunk_shape * int(block_multiplier)
            ) * self.segmentation_idi.voxel_size

        self.mask = None
        if mask_config:
            self.mask = MasksFromConfig(
                mask_config,
                output_voxel_size=self.segmentation_idi.voxel_size,
                connectivity=2,
                caller_scale_factor=self.segmentation_idi.voxel_size_scale_factor,
            )

        self.delete_tmp = delete_tmp

        self.output_idi = create_multiscale_dataset_idi(
            self.output_path,
            dtype=np.float32,
            voxel_size=self.segmentation_idi.voxel_size,
            total_roi=self.roi,
            write_size=self.block_size,
            original_voxel_size=self.segmentation_idi.original_voxel_size,
        )

    @staticmethod
    def calculate_block_edt(
        block_index,
        segmentation_idi: ImageDataInterface,
        output_idi: ImageDataInterface,
        padding_nm: float,
        block_size,
        mask: MasksFromConfig = None,
    ):
        voxel_size = segmentation_idi.voxel_size
        original_voxel_size = segmentation_idi.original_voxel_size
        padding_voxels_per_axis = tuple(
            int(np.ceil(padding_nm / vs)) for vs in original_voxel_size
        )
        padding = Coordinate(
            p * int(vs) for p, vs in zip(padding_voxels_per_axis, voxel_size)
        )

        block = create_block_from_index(
            output_idi, block_index, padding=padding, block_size=block_size
        )

        if mask:
            mask_block_data = mask.process_block(roi=block.write_roi)
            if not np.any(mask_block_data):
                write_shape = tuple(
                    int(round(s / vs)) for s, vs in zip(block.write_roi.shape, voxel_size)
                )
                output_idi.ds[block.write_roi] = np.zeros(write_shape, dtype=np.float32)
                return

        data = segmentation_idi.to_ndarray_ts(block.read_roi)
        if mask:
            mask_data = mask.process_block(roi=block.read_roi)
            data = data * mask_data

        if not np.any(data):
            write_shape = tuple(
                int(round(s / vs)) for s, vs in zip(block.write_roi.shape, voxel_size)
            )
            output_idi.ds[block.write_roi] = np.zeros(write_shape, dtype=np.float32)
            return

        # Multi-label aware: distance to the nearest voxel with a *different*
        # label (background included), computed directly on the labeled data
        # -- not a binarized mask -- so this is exact per-instance EDT, not
        # an organelle-vs-background approximation.
        distance = edt_module.edt(data, anisotropy=tuple(original_voxel_size))
        distance = trim_array_anisotropic(distance, padding, voxel_size)
        output_idi.ds[block.write_roi] = distance.astype(np.float32)

    def calculate_edt(self):
        num_blocks = dask_util.get_num_blocks(
            self.segmentation_idi, roi=self.roi, block_size=self.block_size
        )
        dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            f"computing blockwise EDT for {self.segmentation_path}",
            ComputeEDT.calculate_block_edt,
            self.segmentation_idi,
            self.output_idi,
            self.padding_nm,
            self.block_size,
            self.mask,
        )

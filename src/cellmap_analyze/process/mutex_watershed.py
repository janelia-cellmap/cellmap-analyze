# %%
import numpy as np
from cellmap_analyze.process.connected_components import ConnectedComponents
from cellmap_analyze.process.morphological_operations import MorphologicalOperations
from cellmap_analyze.process.clean_connected_components import CleanConnectedComponents

from cellmap_analyze.util import dask_util
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
)
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.io_util import (
    get_name_from_path,
    get_output_path_from_input_path,
    split_dataset_path,
)

import logging
import os
from cellmap_analyze.util.mask_util import MasksFromConfig
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.zarr_util import create_multiscale_dataset
import cc3d

from scipy import ndimage
import mwatershed as mws
import fastremap
import fastmorph

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MutexWatershed(ComputeConfigMixin):
    def __init__(
        self,
        affinities_path,
        output_path,
        adjacent_edge_bias=-0.4,
        lr_biases=None,
        lr_bias_ratio=None,
        filter_val=0.5,
        neighborhood=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 3],
            [9, 0, 0],
            [0, 9, 0],
            [0, 0, 9],
        ],
        mask_config=None,
        roi=None,
        minimum_volume_nm_3=0,
        maximum_volume_nm_3=np.inf,
        num_workers=10,
        connectivity=2,
        padding_voxels=None,
        do_opening=False,
        delete_tmp=False,
        chunk_shape=None,
    ):
        super().__init__(num_workers)
        self.neighborhood = neighborhood
        self.affinities_idi = ImageDataInterface(affinities_path)
        if chunk_shape is None:
            chunk_shape = self.affinities_idi.chunk_shape[1:]
        # affinities information
        self.neighborhood = neighborhood
        self.adjacent_edge_bias = adjacent_edge_bias
        if lr_biases is None and lr_bias_ratio is None:
            raise ValueError("Either lr_biases or lr_bias_ratio must be provided")
        self.lr_bias = lr_biases if lr_biases else lr_bias_ratio
        self.filter_val = filter_val
        self.padding_voxels = padding_voxels
        self.do_opening = do_opening

        if roi is None:
            self.roi = self.affinities_idi.roi
        else:
            self.roi = roi

        self.voxel_size = self.affinities_idi.voxel_size
        self.do_full_connected_components = False
        output_ds_basepath = split_dataset_path(output_path)[0]
        os.makedirs(output_ds_basepath, exist_ok=True)

        # Use helper function to generate blockwise path (handles root datasets correctly)
        self.connected_components_blockwise_path = get_output_path_from_input_path(
            output_path, "_blockwise"
        )

        # For new zarr files (root datasets), also create parent directory
        if not self.connected_components_blockwise_path.startswith(output_ds_basepath):
            os.makedirs(
                os.path.dirname(self.connected_components_blockwise_path), exist_ok=True
            )

        create_multiscale_dataset(
            self.connected_components_blockwise_path,
            dtype=np.uint64,
            voxel_size=self.voxel_size,
            total_roi=self.roi,
            write_size=np.array(chunk_shape) * self.voxel_size,
        )
        self.connected_components_blockwise_idi = ImageDataInterface(
            self.connected_components_blockwise_path + "/s0",
            mode="r+",
            chunk_shape=chunk_shape,
        )

        self.output_path = str(output_path).rstrip("/")

        # evaluate minimum_volume_nm_3 voxels if it is a string
        if type(minimum_volume_nm_3) == str:
            minimum_volume_nm_3 = float(minimum_volume_nm_3)
        if type(maximum_volume_nm_3) == str:
            maximum_volume_nm_3 = float(maximum_volume_nm_3)
        self.minimum_volume_nm_3 = minimum_volume_nm_3
        self.maximum_volume_nm_3 = maximum_volume_nm_3
        self.minimum_volume_voxels = minimum_volume_nm_3 / np.prod(self.voxel_size)
        self.maximum_volume_voxels = maximum_volume_nm_3 / np.prod(self.voxel_size)

        self.mask_config = mask_config
        self.mask = None
        if self.mask_config:
            self.mask = MasksFromConfig(
                self.mask_config,
                output_voxel_size=self.voxel_size,
                connectivity=connectivity,
            )

        self.connectivity = connectivity
        self.delete_tmp = delete_tmp

    @staticmethod
    def filter_fragments(
        affs_data: np.ndarray, fragments_data: np.ndarray, filter_val: float
    ) -> None:
        """Allows filtering of MWS fragments based on mean value of affinities & fragments. Will filter and update the fragment array in-place.

        Args:
            aff_data (``np.ndarray``):
                An array containing affinity data.

            fragments_data (``np.ndarray``):
                An array containing fragment data.

            filter_val (``float``):
                Threshold to filter if the average value falls below.
        """

        average_affs: float = np.mean(affs_data.data, axis=0)

        filtered_fragments: list = []

        fragment_ids: np.ndarray = np.unique(fragments_data)

        for fragment, mean in zip(
            fragment_ids, ndimage.mean(average_affs, fragments_data, fragment_ids)
        ):
            if mean < filter_val:
                filtered_fragments.append(fragment)

        filtered_fragments: np.ndarray = np.array(
            filtered_fragments, dtype=fragments_data.dtype
        )
        # replace: np.ndarray = np.zeros_like(filtered_fragments)
        fastremap.mask(fragments_data, filtered_fragments, in_place=True)

    @staticmethod
    def mutex_watershed(
        affinities, neighborhood, adjacent_edge_bias, lr_bias, filter_val
    ):
        if affinities.dtype == np.uint8:
            # logger.info("Assuming affinities are in [0,255]")
            max_affinity_value: float = 255.0
            affinities = affinities.astype(np.float64)
        else:
            affinities = affinities.astype(np.float64)
            max_affinity_value: float = 1.0

        affinities /= max_affinity_value

        if affinities.max() < 1e-4:
            segmentation = np.zeros(affinities.shape[1:], dtype=np.uint64)
            return segmentation

        random_noise = np.random.randn(*affinities.shape) * 0.0001
        smoothed_affs = (
            ndimage.gaussian_filter(
                affinities, sigma=(0, *(np.amax(neighborhood, axis=0) / 3))
            )
            - 0.5
        ) * 0.001
        shift = []
        bias_idx = -1
        previous_offset = np.linalg.norm(neighborhood[0])
        for offset in neighborhood:
            current_offset = np.linalg.norm(offset)
            if current_offset <= 1:
                shift.append(adjacent_edge_bias)
            else:
                if type(lr_bias) is list:
                    if current_offset > previous_offset:
                        bias_idx += 1
                    shift.append(lr_bias[bias_idx])
                else:
                    shift.append(current_offset * lr_bias)
            previous_offset = current_offset
        shift = np.array(shift).reshape((-1, *((1,) * (len(affinities.shape) - 1))))

        # filter fragments
        segmentation = mws.agglom(
            affinities + shift + random_noise + smoothed_affs,
            offsets=neighborhood,
        )

        if filter_val > 0.0:
            MutexWatershed.filter_fragments(affinities, segmentation, filter_val)
        # fragment_ids = fastremap.unique(segmentation[segmentation > 0])
        # fastremap.mask_except(segmentation, filtered_fragments, in_place=True)
        fastremap.renumber(segmentation, in_place=True)
        return segmentation

    @staticmethod
    def instancewise_instance_segmentation(segmentation, connectivity, do_opening):
        # ids = fastremap.unique(segmentation[segmentation > 0])
        if do_opening:
            segmentation = fastmorph.erode(segmentation)
        cc = cc3d.connected_components(
            segmentation,
            connectivity=6 + 12 * (connectivity >= 2) + 8 * (connectivity >= 3),
            out_dtype=np.uint64,
        )
        # We do dilation after the fact to prevent overmerging
        return cc

    def calculate_block_connected_components(
        block_index,
        affinities_idi: ImageDataInterface,
        connected_components_blockwise_idi: ImageDataInterface,
        neighborhood,
        adjacent_edge_bias,
        lr_bias,
        filter_val,
        mask: MasksFromConfig = None,
        connectivity=2,
        padding_voxels=None,
        do_opening=True,
    ):
        if not padding_voxels:
            padding_voxels = np.amax(neighborhood)

        padding = padding_voxels * connected_components_blockwise_idi.voxel_size[0]
        # add half a block of padding
        block = create_block_from_index(
            connected_components_blockwise_idi,
            block_index,
            padding=padding,
        )

        if mask:
            mask_block = mask.process_block(roi=block.read_roi)
            if not mask_block.any():
                connected_components_blockwise_idi.ds[block.write_roi] = 0

        affinities = affinities_idi.to_ndarray_ts(block.read_roi)

        if mask:
            affinities *= mask_block[None, :, :, :]

        segmentation = MutexWatershed.mutex_watershed(
            affinities=affinities,
            neighborhood=neighborhood,
            adjacent_edge_bias=adjacent_edge_bias,
            lr_bias=lr_bias,
            filter_val=filter_val,
        )
        segmentation = MutexWatershed.instancewise_instance_segmentation(
            segmentation, connectivity=connectivity, do_opening=do_opening
        )

        global_id_offset = np.uint64(
            block_index
            * np.prod(
                block.full_block_size
                / connected_components_blockwise_idi.voxel_size[0],
                dtype=np.uint64,
            )
        )
        segmentation[segmentation > 0] += global_id_offset
        connected_components_blockwise_idi.ds[block.write_roi] = segmentation[
            padding_voxels:-padding_voxels,
            padding_voxels:-padding_voxels,
            padding_voxels:-padding_voxels,
        ]

    def calculate_connected_components_blockwise(self):
        num_blocks = dask_util.get_num_blocks(
            self.connected_components_blockwise_idi, roi=self.roi
        )
        dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            f"calculating blockwise connected components with mws for {self.affinities_idi.path}",
            MutexWatershed.calculate_block_connected_components,
            self.affinities_idi,
            self.connected_components_blockwise_idi,
            self.neighborhood,
            self.adjacent_edge_bias,
            self.lr_bias,
            self.filter_val,
            self.mask,
            self.connectivity,
            self.padding_voxels,
            self.do_opening,
        )

    def get_connected_components(self):
        self.calculate_connected_components_blockwise()

        # Use helper function for eroded path (handles root datasets correctly)
        eroded_path = (
            get_output_path_from_input_path(self.output_path, "_eroded")
            if self.do_opening
            else self.output_path
        )

        cc = ConnectedComponents(
            connected_components_blockwise_path=self.connected_components_blockwise_path
            + "/s0",
            output_path=eroded_path,
            num_workers=self.num_workers,
            minimum_volume_nm_3=0 if self.do_opening else self.minimum_volume_nm_3,
            maximum_volume_nm_3=np.inf if self.do_opening else self.maximum_volume_nm_3,
            delete_tmp=self.delete_tmp,
            connectivity=self.connectivity,
            mask_config=self.mask_config,
        )
        cc.merge_connected_components_across_blocks()

        if self.do_opening:
            # Use helper function for dilated path (handles root datasets correctly)
            eroded_dilated_path = get_output_path_from_input_path(
                self.output_path, "_eroded_dilated"
            )

            mo = MorphologicalOperations(
                input_path=eroded_path + "/s0",
                output_path=eroded_dilated_path,
                operation="dilation",
                iterations=1,
                num_workers=self.num_workers,
                connectivity=self.connectivity,
                mask_config=self.mask_config,
            )
            mo.perform_morphological_operation()

            ccc = CleanConnectedComponents(
                input_path=eroded_dilated_path + "/s0",
                output_path=self.output_path,
                num_workers=self.num_workers,
                minimum_volume_nm_3=self.minimum_volume_nm_3,
                maximum_volume_nm_3=self.maximum_volume_nm_3,
                connectivity=self.connectivity,
                delete_tmp=self.delete_tmp,
                mask_config=self.mask_config,
            )
            ccc.clean_connected_components()
            if self.delete_tmp:
                dask_util.delete_tmp_dir_blockwise(
                    eroded_path + "/s0", self.num_workers, self.compute_args
                )
                dask_util.delete_tmp_dir_blockwise(
                    eroded_dilated_path + "/s0",
                    self.num_workers,
                    self.compute_args,
                )

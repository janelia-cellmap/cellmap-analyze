import numpy as np
from cellmap_analyze.process.connected_components import ConnectedComponents
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
)
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.io_util import get_output_path_from_input_path
import logging
import itertools
from cellmap_analyze.util.mask_util import MasksFromConfig
import fastremap

from cellmap_analyze.util.mixins import ComputeConfigMixin
import shutil

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class CleanConnectedComponents(ComputeConfigMixin):
    def __init__(
        self,
        input_path,
        output_path=None,
        mask_config=None,
        roi=None,
        minimum_volume_nm_3=0,
        maximum_volume_nm_3=np.inf,
        num_workers=10,
        mask_connectivity=2,
        connectivity=2,
        fill_holes=False,
        delete_tmp=True,
        chunk_shape=None,
    ):
        super().__init__(num_workers)

        self.input_path = input_path
        self.input_idi = ImageDataInterface(self.input_path, chunk_shape=chunk_shape)
        if roi is None:
            self.roi = self.input_idi.roi
        else:
            self.roi = roi

        self.voxel_size = self.input_idi.voxel_size

        self.output_path = output_path
        if self.output_path is None:
            self.output_path = get_output_path_from_input_path(
                self.input_path, "_cleaned"
            )
        self.output_path = str(self.output_path).rstrip("/")

        # Use helper function to generate relabeling dict path (handles root datasets correctly)
        self.relabeling_dict_path = get_output_path_from_input_path(
            self.output_path, "_relabeling_dict"
        ) + "/"

        # evaluate minimum_volume_nm_3 voxels if it is a string
        if type(minimum_volume_nm_3) == str:
            minimum_volume_nm_3 = float(minimum_volume_nm_3)
        if type(maximum_volume_nm_3) == str:
            maximum_volume_nm_3 = float(maximum_volume_nm_3)

        self.minimum_volume_voxels = minimum_volume_nm_3 / np.prod(self.voxel_size)
        self.maximum_volume_voxels = maximum_volume_nm_3 / np.prod(self.voxel_size)

        self.mask = None
        if mask_config:
            self.mask = MasksFromConfig(
                mask_config,
                output_voxel_size=self.voxel_size,
                connectivity=mask_connectivity,
            )

        self.connectivity = connectivity
        self.fill_holes = fill_holes
        self.delete_tmp = delete_tmp

    @staticmethod
    def volume_filter_connected_ids(
        connected_ids, id_to_volume_dict, minimum_volume_voxels
    ):
        kept_ids = []
        removed_ids = []
        for current_connected_ids in connected_ids:
            volume = sum([id_to_volume_dict[id] for id in current_connected_ids])
            if volume >= minimum_volume_voxels:
                kept_ids.append(current_connected_ids)
            else:
                removed_ids.append(current_connected_ids)
        return kept_ids, removed_ids

    @staticmethod
    def get_connected_component_information_blockwise(
        block_index,
        connected_components_blockwise_idi: ImageDataInterface,
        mask: MasksFromConfig = None,
    ):
        try:
            block = create_block_from_index(
                connected_components_blockwise_idi,
                block_index,
            )
            data = connected_components_blockwise_idi.to_ndarray_ts(
                block.read_roi,
            )
            # need to get these premask since during relabeling we have to assing em to zero
            unique_ids = fastremap.unique(data)
            if mask:
                mask_block = mask.process_block(roi=block.read_roi)
                data *= mask_block

            # get information only from actual block(not including padding)
            id_to_volume_dict = ConnectedComponents.get_object_sizes(data)
            block.relabeling_dict = {id: 0 for id in unique_ids}
        except Exception as e:
            raise Exception(
                f"Error {e} in get_connected_component_information_blockwise {block_index}, {connected_components_blockwise_idi.voxel_size}"
            )
        return id_to_volume_dict, set()

    def get_connected_component_information(self):
        num_blocks = dask_util.get_num_blocks(self.input_idi, self.roi)
        self.id_to_volume_dict, _ = dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            f"getting blockwise connected component information during cleaning for {self.input_idi.path}",
            CleanConnectedComponents.get_connected_component_information_blockwise,
            self.input_idi,
            self.mask,
            merge_info=(
                ConnectedComponents._merge_tuples,
                get_output_path_from_input_path(
                    self.output_path, "_tmp_cleaned_connected_component_info_to_merge"
                ),
            ),
        )

    def get_final_connected_components(self):
        # make it a list of list to be consistence with connectedcomponents volume filter
        old_ids = [[id] for id in self.id_to_volume_dict.keys()]
        if self.minimum_volume_voxels > 0 or self.maximum_volume_voxels < np.inf:
            with io_util.TimingMessager("Volume filter connected", logger):
                old_ids, _ = ConnectedComponents.volume_filter_connected_ids(
                    old_ids,
                    self.id_to_volume_dict,
                    self.minimum_volume_voxels,
                    self.maximum_volume_voxels,
                )

        del self.id_to_volume_dict

        # sort connected_ids by the minimum id in each connected component
        new_ids = list(range(1, len(old_ids) + 1))
        old_ids = list(itertools.chain(*old_ids))

        if len(new_ids) == 0:
            self.new_dtype = np.uint8
            relabeling_dict = {}
        else:
            self.new_dtype = np.min_scalar_type(max(new_ids))
            relabeling_dict = dict(zip(old_ids, new_ids))

        ConnectedComponents.write_memmap_relabeling_dicts(
            relabeling_dict, self.relabeling_dict_path
        )

    def clean_connected_components(self):
        # get blockwise connected component information
        self.get_connected_component_information()
        # get final connected components necessary for relabeling, including volume filtering
        self.get_final_connected_components()

        ConnectedComponents.relabel_dataset(
            self.input_idi,
            self.output_path,
            self.roi,
            self.new_dtype,
            self.relabeling_dict_path,
            self.num_workers,
            self.compute_args,
            mask=self.mask,
        )

        if self.fill_holes:
            from .fill_holes import FillHoles

            # Use helper function for filled path (handles root datasets correctly)
            filled_path = get_output_path_from_input_path(self.output_path, "_filled")

            fh = FillHoles(
                input_path=self.output_path + "/s0",
                output_path=filled_path,
                num_workers=self.num_workers,
                roi=self.roi,
                connectivity=self.connectivity,
            )
            fh.fill_holes()

        shutil.rmtree(self.relabeling_dict_path)

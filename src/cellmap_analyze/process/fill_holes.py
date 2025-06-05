import numpy as np
from cellmap_analyze.util import dask_util
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
)
from cellmap_analyze.util.image_data_interface import ImageDataInterface

import logging
import fastremap
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.zarr_util import create_multiscale_dataset_idi
from skimage.segmentation import find_boundaries
from .connected_components import ConnectedComponents
import shutil

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FillHoles(ComputeConfigMixin):
    def __init__(
        self,
        input_path,
        output_path=None,
        num_workers=10,
        connectivity=2,
        roi=None,
        chunk_shape=None,
    ):
        super().__init__(num_workers)
        self.input_idi = ImageDataInterface(input_path, chunk_shape=chunk_shape)
        self.roi = roi
        if self.roi is None:
            self.roi = self.input_idi.roi
        self.voxel_size = self.input_idi.voxel_size
        self.connectivity = connectivity

        self.output_path = output_path
        if self.output_path is None:
            self.output_path = input_path + "_filled"
        self.relabeling_dict_path = self.output_path + "_relabeling_dict/"
        self.holes_path = self.output_path + "_holes"

    def get_hole_information_blockwise(
        block_index,
        input_idi: ImageDataInterface,
        holes_idi: ImageDataInterface,
        connectivity,
    ):
        block = create_block_from_index(
            input_idi,
            block_index,
            padding=input_idi.voxel_size,
        )
        hole_to_object_dict = {}
        roi = block.read_roi.grow(input_idi.voxel_size, input_idi.voxel_size)
        holes = holes_idi.to_ndarray_ts(roi)
        input = input_idi.to_ndarray_ts(roi)

        input_boundaries = find_boundaries(input, mode="inner").astype(np.uint64)
        hole_boundaries = find_boundaries(holes, mode="inner").astype(np.uint64)

        max_input_id = np.max(input)
        holes = holes.astype(np.uint64)
        holes[holes > 0] += max_input_id  # so there is no id overlap
        data = holes + input
        mask = np.logical_or(input_boundaries, hole_boundaries)
        touching_ids = ConnectedComponents.get_touching_ids(data, mask, connectivity)

        for id1, id2 in touching_ids:
            if id2 <= max_input_id:
                continue
            id2 -= max_input_id
            # then input objects are touching holes
            object_ids = hole_to_object_dict.get(id2, set())
            object_ids.add(id1)
            hole_to_object_dict[id2] = object_ids

        return hole_to_object_dict

    @staticmethod
    def _merge_hole_partials(
        hole_dict_acc: dict[int, set[int]],
        hole_dict_2: dict[int, set[int]],
    ) -> dict[int, set[int]]:
        """Merge two dictionaries of hole IDs to sets of object IDs.
        If a hole ID appears in both dictionaries, the sets of object IDs are
        combined. If a hole ID appears in only one dictionary, its set of object
        IDs is retained.
        If the accumulator is None, it initializes an empty dictionary.
        Args:
            hole_dict_acc (dict[int, set[int]]): The accumulator dictionary.
            hole_dict2 (dict[int, set[int]]): The dictionary to merge into the accumulator.
        Returns:
            dict[int, set[int]]: The merged dictionary of hole IDs to sets of object IDs.
        """

        if hole_dict_acc is None:
            hole_dict_acc = {}

        merged_hole_dict = hole_dict_acc.copy()
        for hole_id, obj_set in hole_dict_2.items():
            merged_hole_dict[hole_id] = merged_hole_dict.get(hole_id, set()).union(
                obj_set
            )

        return merged_hole_dict

    @staticmethod
    def __postprocess_hole_dict(raw_hole_dict: dict[int, set[int]]) -> dict[int, int]:
        """
        For each hole_id, if it touches >1 objects, assign 0.
        Otherwise, extract the single object ID.
        """
        final = {}
        for hole_id, obj_set in raw_hole_dict.items():
            if len(obj_set) > 1:
                final[hole_id] = 0
            else:
                final[hole_id] = next(iter(obj_set))
        return final

    def get_hole_assignments(self):
        num_blocks = dask_util.get_num_blocks(self.input_idi)
        hole_to_object_dict = dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            "calculating hole information",
            FillHoles.get_hole_information_blockwise,
            input_idi=self.input_idi,
            holes_idi=self.holes_idi,
            connectivity=self.connectivity,
            merge_fn=FillHoles._merge_hole_partials,
            merge_identity=None,
        )

        hole_to_object_dict = FillHoles.__postprocess_hole_dict(hole_to_object_dict)

        ConnectedComponents.write_memmap_relabeling_dicts(
            hole_to_object_dict,
            self.relabeling_dict_path,
        )

    @staticmethod
    def relabel_block(
        block_index, input_idi, holes_idi, output_idi, relabeling_dict_path
    ):
        # read block from pickle file
        block = create_block_from_index(input_idi, block_index)

        # print_with_datetime(block.relabeling_dict, logger)
        input = input_idi.to_ndarray_ts(
            block.write_roi,
        )
        holes = holes_idi.to_ndarray_ts(block.write_roi)
        hole_ids = fastremap.unique(holes[holes > 0])
        relabeling_dict = ConnectedComponents.get_updated_relabeling_dict(
            hole_ids, relabeling_dict_path
        )
        if len(relabeling_dict) > 0:
            fastremap.remap(
                holes,
                relabeling_dict,
                preserve_missing_labels=True,
                in_place=True,
            )
        output_idi.ds[block.write_roi] = input + holes

    def relabel_dataset(self):
        self.output_idi = create_multiscale_dataset_idi(
            self.output_path,
            dtype=self.input_idi.ds.dtype,
            voxel_size=self.voxel_size,
            total_roi=self.roi,
            write_size=self.input_idi.chunk_shape * self.input_idi.voxel_size,
        )

        num_blocks = dask_util.get_num_blocks(self.input_idi, roi=self.roi)
        dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            "relabeling dataset",
            FillHoles.relabel_block,
            self.input_idi,
            self.holes_idi,
            self.output_idi,
            self.relabeling_dict_path,
        )

    def fill_holes(self):

        # do connected components for holes
        cc = ConnectedComponents(
            input_path=self.input_idi.path,
            output_path=self.holes_path,
            num_workers=self.num_workers,
            connectivity=self.connectivity,
            invert=True,
            calculating_holes=True,
            roi=self.roi,
        )
        cc.get_connected_components()
        self.holes_idi = ImageDataInterface(
            self.holes_path + "/s0", mode="r+", chunk_shape=self.input_idi.chunk_shape
        )
        # get the assignments of holes to objects or background
        self.get_hole_assignments()
        self.relabel_dataset()
        dask_util.delete_tmp_zarr(
            self.holes_idi,
            self.num_workers,
            self.compute_args,
        )
        dask_util.delete_tmp_zarr(
            self.holes_path + "_blockwise/s0",
            self.num_workers,
            self.compute_args,
        )

        shutil.rmtree(self.relabeling_dict_path)

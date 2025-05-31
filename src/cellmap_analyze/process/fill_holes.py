import pickle
import types
import numpy as np
from tqdm import tqdm
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.dask_util import (
    DaskBlock,
    create_block_from_index,
    dask_computer,
    guesstimate_npartitions,
)
from cellmap_analyze.util.image_data_interface import ImageDataInterface

import logging
import dask.bag as db
import fastremap
import os
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.zarr_util import create_multiscale_dataset_idi
from skimage.segmentation import find_boundaries
from .connected_components import ConnectedComponents

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
        minimum_volume_nm_3=0,
        num_workers=10,
        connectivity=2,
        delete_tmp=False,
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
        block.hole_to_object_dict = {}
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
            object_ids = block.hole_to_object_dict.get(id2, set())
            object_ids.add(id1)
            block.hole_to_object_dict[id2] = object_ids

        return block

    @staticmethod
    def __combine_hole_information(blocks):
        # 1) Merge hole_to_object_dict from each block
        hole_to_object_dict = {}
        for idx, block in enumerate(tqdm(blocks, desc="Merging hole info")):
            # The first block just seeds our dictionary (make a copy to avoid mutating original)
            if idx == 0:
                hole_to_object_dict = block.hole_to_object_dict.copy()
                continue

            # For subsequent blocks, union any sets in their dict into ours
            for hole_id, obj_set in block.hole_to_object_dict.items():
                existing = hole_to_object_dict.get(hole_id, set())
                hole_to_object_dict[hole_id] = existing.union(obj_set)

        # 5) Post-process: if a hole is touching more than one object, mark it 0; otherwise pop the single ID
        for hole_id, object_ids in list(hole_to_object_dict.items()):
            if len(object_ids) > 1:
                hole_to_object_dict[hole_id] = 0
            else:
                hole_to_object_dict[hole_id] = object_ids.pop()

        return blocks, hole_to_object_dict

    def get_final_hole_assignments(self, hole_to_object_dict):
        # update blockwise relabeing dicts
        for block in self.blocks:
            block.relabeling_dict = {
                id: hole_to_object_dict.get(id, 0)
                for id in block.hole_to_object_dict.keys()
            }
            block.hole_to_object_dict = None

    def get_hole_assignments(self):
        num_blocks = dask_util.get_num_blocks(self.input_idi)
        bagged_results = dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            "calculating hole information",
            FillHoles.get_hole_information_blockwise,
            input_idi=self.input_idi,
            holes_idi=self.holes_idi,
            connectivity=self.connectivity,
        )
        # moved this out of dask, seems fast enough without having to daskify
        with io_util.Timing_Messager("Combining bagged results", logger):
            blocks, hole_to_object_dict = FillHoles.__combine_hole_information(
                bagged_results
            )
        # get blockwise dict assignments
        self.blocks = [None] * num_blocks
        for block in blocks:
            self.blocks[block.index] = block
            block.relabeling_dict = {
                id: hole_to_object_dict.get(id, 0)
                for id in block.hole_to_object_dict.keys()
            }
            block.hole_to_object_dict = None

    @staticmethod
    def relabel_block(
        block_index,
        input_idi,
        holes_idi,
        output_idi,
    ):
        # read block from pickle file
        block_coords = create_block_from_index(input_idi, block_index).coords
        block_coords_string = "/".join([str(c) for c in block_coords])
        with open(f"{holes_idi.path}/{block_coords_string}.pkl", "rb") as handle:
            block = pickle.load(handle)
        # print_with_datetime(block.relabeling_dict, logger)
        input = input_idi.to_ndarray_ts(
            block.write_roi,
        )
        holes = holes_idi.to_ndarray_ts(block.write_roi)

        if len(block.relabeling_dict) > 0:
            fastremap.remap(
                holes,
                block.relabeling_dict,
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

        num_blocks = len(self.blocks)
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

        # relabel dataset
        ConnectedComponents.write_out_block_objects(
            path=self.holes_path + "/s0",
            blocks=self.blocks,
            num_local_threads_available=self.num_local_threads_available,
            local_config=self.local_config,
            compute_args=self.compute_args,
        )
        self.relabel_dataset()
        dask_util.delete_tmp_dataset(
            self.holes_path + "/s0",
            self.blocks,
            self.num_workers,
            self.compute_args,
        )
        dask_util.delete_tmp_dataset(
            self.holes_path + "_blockwise/s0",
            self.blocks,
            self.num_workers,
            self.compute_args,
        )

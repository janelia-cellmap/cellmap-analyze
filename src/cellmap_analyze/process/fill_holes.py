from collections import defaultdict
from typing import Set
import numpy as np
from cellmap_analyze.util import dask_util
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
)
from cellmap_analyze.util.image_data_interface import ImageDataInterface

import logging
import fastremap
from cellmap_analyze.util.io_util import get_output_path_from_input_path
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.cythonizing.touching import get_touching_ids
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

        if output_path is None:
            output_path = get_output_path_from_input_path(input_path, "_filled")
        self.output_path = str(output_path).rstrip("/")

        # Use helper function to generate auxiliary paths (handles root datasets correctly)
        self.relabeling_dict_path = get_output_path_from_input_path(
            self.output_path, "_relabeling_dict"
        ) + "/"
        self.holes_path = get_output_path_from_input_path(self.output_path, "_holes")

    def get_hole_information_blockwise(
        block_index,
        input_idi: ImageDataInterface,
        holes_idi: ImageDataInterface,
        connectivity,
    ):
        # pad by two pixels since to determine boundary need an extra pixel
        block = create_block_from_index(
            input_idi,
            block_index,
            padding=input_idi.voxel_size,
            padding_direction="neg_with_edge_pos",
        )
        hole_to_object_dict = {}
        holes = holes_idi.to_ndarray_ts(block.read_roi)
        input = input_idi.to_ndarray_ts(block.read_roi)

        input_boundaries = find_boundaries(
            input, mode="inner", connectivity=connectivity
        ).astype(np.uint64)
        hole_boundaries = find_boundaries(
            holes, mode="inner", connectivity=connectivity
        ).astype(np.uint64)

        max_input_id = np.max(input).astype(np.uint64)
        holes = holes.astype(np.uint64)
        holes[holes > 0] += max_input_id  # so there is no id overlap
        data = holes + input
        mask = np.logical_or(input_boundaries, hole_boundaries)
        touching_ids = get_touching_ids(data, mask, connectivity)

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
    def _merge_hole_to_object_dicts(hole_to_object_dicts) -> dict[int, set[int]]:
        hole_to_object_dict: defaultdict[int, Set[int]] = defaultdict(set)
        for d in hole_to_object_dicts:
            for hole_id, touching_ids in d.items():
                hole_to_object_dict[hole_id].update(touching_ids)
        return hole_to_object_dict

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
            f"calculating blockwise hole information for {self.input_idi.path}",
            FillHoles.get_hole_information_blockwise,
            input_idi=self.input_idi,
            holes_idi=self.holes_idi,
            connectivity=self.connectivity,
            merge_info=(
                FillHoles._merge_hole_to_object_dicts,
                get_output_path_from_input_path(
                    self.output_path, "_tmp_hole_objects_to_dict_to_merge"
                ),
            ),
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
            if input.dtype.itemsize > holes.dtype.itemsize:
                holes = holes.astype(input.dtype)
            fastremap.remap(
                holes,
                relabeling_dict,
                preserve_missing_labels=True,
                in_place=True,
            )

        output_idi.ds[block.write_roi] = input + holes.astype(input.dtype)

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
            f"relabeling dataset with holes filled to {self.output_idi.path}",
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
        dask_util.delete_tmp_dir_blockwise(
            self.holes_idi,
            self.num_workers,
            self.compute_args,
        )

        # Use helper function to get blockwise path (handles root datasets correctly)
        holes_blockwise_path = get_output_path_from_input_path(
            self.holes_path, "_blockwise"
        )
        dask_util.delete_tmp_dir_blockwise(
            holes_blockwise_path + "/s0",
            self.num_workers,
            self.compute_args,
        )

        shutil.rmtree(self.relabeling_dict_path)

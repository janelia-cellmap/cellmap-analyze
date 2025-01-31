from collections import defaultdict
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
from cellmap_analyze.util.io_util import (
    get_name_from_path,
    split_dataset_path,
)

import logging
from skimage.graph import pixel_graph
import networkx as nx
import dask.bag as db
import itertools
from funlib.segment.arrays import replace_values
import os
from cellmap_analyze.util.zarr_util import create_multiscale_dataset
import skimage
from skimage.segmentation import find_boundaries
from .connected_components import ConnectedComponents

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FillHoles:
    def __init__(
        self,
        input_path,
        output_path=None,
        minimum_volume_nm_3=0,
        num_workers=10,
        connectivity=2,
        delete_tmp=False,
        roi=None,
    ):
        self.input_idi = ImageDataInterface(input_path)
        self.roi = roi
        if self.roi is None:
            self.roi = self.input_idi.roi
        self.voxel_size = self.input_idi.voxel_size
        self.connectivity = connectivity

        self.output_path = output_path
        if self.output_path is None:
            self.output_path = input_path + "_filled"

        self.holes_path = self.output_path + "_holes"
        # holes_blockwise_path = holes_path + "_blockwise"

        # for hole_ds in [holes_path, holes_blockwise_path]:
        #     create_multiscale_dataset(
        #         hole_ds,
        #         dtype=np.uint64,
        #         voxel_size=self.voxel_size,
        #         total_roi=self.roi,
        #         write_size=self.input_idi.chunk_shape * self.voxel_size,
        #     )
        # self.holes_idi = ImageDataInterface(holes_path + "/s0", mode="r+")
        # self.holes_blockwise_idi = ImageDataInterface(
        #     holes_blockwise_path + "/s0", mode="r+"
        # )

        self.num_workers = num_workers
        self.compute_args = {}
        if self.num_workers == 1:
            self.compute_args = {"scheduler": "single-threaded"}
            self.num_local_threads_available = 1
            self.local_config = None
        else:
            self.num_local_threads_available = len(os.sched_getaffinity(0))
            self.local_config = {
                "jobqueue": {
                    "local": {
                        "ncpus": self.num_local_threads_available,
                        "processes": self.num_local_threads_available,
                        "cores": self.num_local_threads_available,
                        "log-directory": "job-logs",
                        "name": "dask-worker",
                    }
                }
            }

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

        max_input_id = np.max(input)
        holes[holes > 0] = holes[holes > 0] + max_input_id  # so there is no id overlap

        input_boundaries = find_boundaries(input, mode="inner").astype(np.uint64)
        hole_boundaries = find_boundaries(holes, mode="inner").astype(np.uint64)

        data = holes + input
        mask = np.logical_or(input_boundaries, hole_boundaries)
        touching_ids = ConnectedComponents.get_touching_ids(data, mask, connectivity)

        for id1, id2 in touching_ids:
            if id2 <= max_input_id:
                continue
            id2 -= max_input_id
            # then input objects are touching holes
            object_ids = block.hole_to_object_dict.get(id2, set())
            objectsIds = object_ids.add(id1)
            block.hole_to_object_dict[id2] = object_ids

        return block

    @staticmethod
    def __combine_hole_information(blocks):
        if type(blocks) is DaskBlock:
            blocks = list(blocks)
        if isinstance(blocks, types.GeneratorType):
            blocks = list(blocks)

        hole_to_object_dict = {}
        for idx, block in enumerate(tqdm(blocks)):
            if idx == 0:
                hole_to_object_dict = block.hole_to_object_dict
                continue

            for key, value in block.hole_to_object_dict.items():
                current_values = hole_to_object_dict.get(key, set())
                hole_to_object_dict[key] = current_values.union(value)

        for hole_id, object_ids in hole_to_object_dict.items():
            if len(object_ids) > 1:
                # then is touching two ids so is not a hole we can fill, so delete it from dictionary
                hole_to_object_dict[hole_id] = 0
            else:
                hole_to_object_dict[hole_id] = (
                    object_ids.pop()
                )  # get only element of the set

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
        block_indexes = list(range(num_blocks))
        b = db.from_sequence(
            block_indexes,
            npartitions=guesstimate_npartitions(block_indexes, self.num_workers),
        ).map(
            FillHoles.get_hole_information_blockwise,
            input_idi=self.input_idi,
            holes_idi=self.holes_idi,
            connectivity=self.connectivity,
        )

        with dask_util.start_dask(
            self.num_workers,
            "calculate hole information",
            logger,
        ):
            with io_util.Timing_Messager("Calculating hole information", logger):
                bagged_results = dask_computer(b, self.num_workers, **self.compute_args)

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
        block_coords,
        input_idi,
        holes_idi,
        output_idi,
    ):
        # read block from pickle file
        block_coords_string = "/".join([str(c) for c in block_coords])
        with open(f"{holes_idi.path}/{block_coords_string}.pkl", "rb") as handle:
            block = pickle.load(handle)
        # print_with_datetime(block.relabeling_dict, logger)
        input = input_idi.to_ndarray_ts(
            block.write_roi,
        )
        holes = holes_idi.to_ndarray_ts(block.write_roi)

        if len(block.relabeling_dict) > 0:
            # couldnt do inplace for large uint types because it was converting to floats
            relabeled = np.zeros_like(holes)
            keys, values = zip(*block.relabeling_dict.items())
            replace_values(holes, list(keys), list(values), out_array=relabeled)
            holes = relabeled
        output_idi.ds[block.write_roi] = input + holes

    def relabel_dataset(self):
        create_multiscale_dataset(
            self.output_path,
            dtype=self.input_idi.ds.dtype,
            voxel_size=self.voxel_size,
            total_roi=self.roi,
            write_size=self.input_idi.chunk_shape * self.input_idi.voxel_size,
        )
        self.output_idi = ImageDataInterface(self.output_path + "/s0", mode="r+")

        block_coords = [block.coords for block in self.blocks]
        b = db.from_sequence(
            block_coords,
            npartitions=guesstimate_npartitions(self.blocks, self.num_workers),
        ).map(
            FillHoles.relabel_block,
            self.input_idi,
            self.holes_idi,
            self.output_idi,
        )

        with dask_util.start_dask(self.num_workers, "relabel dataset", logger):
            with io_util.Timing_Messager("Relabeling dataset", logger):
                dask_computer(b, self.num_workers, **self.compute_args)

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
        self.holes_idi = ImageDataInterface(self.holes_path + "/s0", mode="r+")
        # get the assignments of holes to objects or background
        self.get_hole_assignments()

        # relabel dataset
        ConnectedComponents.write_out_block_objects(
            path=self.holes_path + "/s0",
            blocks=self.blocks,
            num_local_threads_available=self.num_local_threads_available,
            local_config=self.local_config,
            num_workers=self.num_workers,
            compute_args=self.compute_args,
        )
        self.relabel_dataset()


# # %%
# FillHoles()
# # %%
# import numpy as np
# import matplotlib.pyplot as plt

# labels = np.array(
#     [
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
#         [0, 0, 1, 1, 1, 5, 0, 5, 0, 0],
#         [0, 0, 1, 1, 1, 5, 0, 5, 0, 0],
#         [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
#         [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     ],
#     dtype=np.uint8,
# )
# from skimage.segmentation import find_boundaries

# find_boundaries(labels, mode="inner").astype(np.uint8)
# # %%

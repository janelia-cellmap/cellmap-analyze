from collections import defaultdict
from typing import Set
import numpy as np
import worldarray as wa
from cellmap_analyze.util import dask_util
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
)
from cellmap_analyze.util.grid_util import crop_to_roi, grid_of

import logging
import uuid
import fastremap
from cellmap_analyze.util.io_util import get_output_path_from_input_path
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.cythonizing.touching import get_touching_ids
from cellmap_analyze.util.zarr_util import create_multiscale_dataset, open_dataset
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
        self.input_vol = open_dataset(input_path)
        self.roi = roi
        # the holes are found on this grid (see fill_holes), and everything is read onto it
        self.grid = crop_to_roi(grid_of(self.input_vol, chunk_shape), roi)
        self.voxel_size = self.input_vol.voxel_size
        self.connectivity = connectivity
        # Per-instance suffix so concurrent runs sharing output_path don't
        # collide on the fixed-name merge dir.
        self._run_id = uuid.uuid4().hex[:8]

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
        grid: wa.Grid,
        input_vol: wa.WorldArray,
        holes_vol: wa.WorldArray,
        connectivity,
    ):
        # pad by a voxel since to determine boundary need an extra pixel
        block = create_block_from_index(
            grid,
            block_index,
            padding=1,
            padding_direction="neg_with_edge_pos",
        )
        hole_to_object_dict = {}
        holes = holes_vol.read_onto(grid, block.read_begin, block.read_end)
        input = input_vol.read_onto(grid, block.read_begin, block.read_end)

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
        num_blocks = dask_util.get_num_blocks(self.grid)
        hole_to_object_dict = dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            f"calculating blockwise hole information for {self.input_vol.level_path}",
            FillHoles.get_hole_information_blockwise,
            grid=self.grid,
            input_vol=self.input_vol,
            holes_vol=self.holes_vol,
            connectivity=self.connectivity,
            merge_info=(
                FillHoles._merge_hole_to_object_dicts,
                get_output_path_from_input_path(
                    self.output_path,
                    f"_tmp_hole_objects_to_dict_to_merge_{self._run_id}",
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
        block_index, grid, input_vol, holes_vol, output_vol, relabeling_dict_path
    ):
        block = create_block_from_index(grid, block_index)

        input = input_vol.read_onto(grid, block.begin, block.end)
        holes = holes_vol.read_onto(grid, block.begin, block.end)
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

        output_vol.write_index(block.begin, block.end, input + holes.astype(input.dtype))

    def relabel_dataset(self):
        self.output_vol = create_multiscale_dataset(
            self.output_path, dtype=self.input_vol.dtype, grid=self.grid
        )

        num_blocks = dask_util.get_num_blocks(self.grid)
        dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            f"relabeling dataset with holes filled to {self.output_vol.level_path}",
            FillHoles.relabel_block,
            self.grid,
            self.input_vol,
            self.holes_vol,
            self.output_vol,
            self.relabeling_dict_path,
        )

    def fill_holes(self):

        # do connected components for holes
        cc = ConnectedComponents(
            input_path=self.input_vol.level_path,
            output_path=self.holes_path,
            num_workers=self.num_workers,
            connectivity=self.connectivity,
            invert=True,
            calculating_holes=True,
            roi=self.roi,
        )
        cc.get_connected_components()
        self.holes_vol = open_dataset(self.holes_path + "/s0", mode="r+")
        # get the assignments of holes to objects or background
        self.get_hole_assignments()
        self.relabel_dataset()
        dask_util.delete_tmp_dir_blockwise(
            self.holes_vol,
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

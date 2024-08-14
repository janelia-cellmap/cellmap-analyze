from funlib.persistence import Array, open_ds
from funlib.geometry import Roi
import numpy as np
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.dask_util import DaskBlock, create_blocks
from cellmap_analyze.util.io_util import (
    get_name_from_path,
    open_ds_tensorstore,
    split_dataset_path,
    to_ndarray_tensorstore,
)
from skimage import measure
from skimage.segmentation import expand_labels, find_boundaries
from sklearn.metrics.pairwise import pairwise_distances
from cellmap_analyze.util.bresenham3D import bresenham3DWithMask
import logging
from skimage.graph import pixel_graph
import networkx as nx
import dask.bag as db
import itertools
from funlib.segment.arrays import replace_values
import os
from cellmap_analyze.util.zarr_util import create_multiscale_dataset

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ConnectedComponents:
    def __init__(
        self,
        tmp_blockwise_ds_path,
        output_ds_path,
        roi=None,
        minimum_volume_nm_3=None,
        num_workers=10,
        connectivity=2,
    ):
        self.tmp_blockwise_ds_path = tmp_blockwise_ds_path
        self.tmp_blockwise_ds = open_ds(*split_dataset_path(self.tmp_blockwise_ds_path))
        self.tmp_blockwise_ds_tensorstore = open_ds_tensorstore(
            self.tmp_blockwise_ds_path
        )
        self.output_ds_path = output_ds_path

        if roi is None:
            self.roi = self.tmp_blockwise_ds.roi
        else:
            self.roi = roi
        self.voxel_size = self.tmp_blockwise_ds.voxel_size
        self.minimum_volume_voxels = minimum_volume_nm_3 / np.prod(self.voxel_size)
        self.connectivity = connectivity
        self.num_workers = num_workers

    @staticmethod
    def convert_position_to_global_id(position, dimensions):
        id = (
            dimensions[0] * dimensions[1] * position[2]
            + dimensions[0] * position[1]
            + position[0]
            + 1
        )
        return id

    @staticmethod
    def get_touching_ids(data, mask, connectivity=2):
        # https://stackoverflow.com/questions/72452267/finding-identity-of-touching-labels-objects-masks-in-images-using-python
        g, nodes = pixel_graph(
            data,
            mask=mask,
            connectivity=connectivity,
        )

        coo = g.tocoo()
        center_coords = nodes[coo.row]
        neighbor_coords = nodes[coo.col]

        center_values = data.ravel()[center_coords]
        neighbor_values = data.ravel()[neighbor_coords]

        # sort to have lowest pair first
        touching_ids = np.sort(
            np.stack([center_values, neighbor_values], axis=1), axis=1
        )
        touching_ids = touching_ids[touching_ids[:, 0] != touching_ids[:, 1]]
        # convert touching_ids to a set of tuples
        touching_ids = set(map(tuple, touching_ids))

        return touching_ids

    @staticmethod
    def get_object_sizes(data):
        labels, counts = np.unique(data[data > 0], return_counts=True)
        return dict(zip(labels, counts))

    @staticmethod
    def get_connected_ids(nodes, edges):
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        connected_ids = list(nx.connected_components(G))
        connected_ids = sorted(connected_ids, key=min)
        return connected_ids

    @staticmethod
    def volume_filter_connected_ids(
        connected_ids, id_to_volume_dict, minimum_volume_voxels
    ):
        kept_ids = []
        removed_ids = []
        for current_connected_ids in connected_ids:
            volume = sum([id_to_volume_dict[id] for id in current_connected_ids])
            if volume > minimum_volume_voxels:
                kept_ids.append(current_connected_ids)
            else:
                removed_ids.append(current_connected_ids)
        return kept_ids, removed_ids

    def get_connected_component_information_blockwise(self, block: DaskBlock):
        data = to_ndarray_tensorstore(
            self.tmp_blockwise_ds_tensorstore,
            block.read_roi,
            self.voxel_size,
            self.tmp_blockwise_ds.roi.offset,
        )
        # if roi.shape != self.ds.intersect(roi).shape:
        #     data = self.ds.to_ndarray(roi, fill_value=0)
        # else:
        #     data = self.ds.to_ndarray(roi)

        mask = data.astype(bool)
        mask[2:, 2:, 2:] = False
        touching_ids = ConnectedComponents.get_touching_ids(
            data, mask=mask, connectivity=self.connectivity
        )

        # get information only from actual block(not including padding)
        id_to_volume_dict = ConnectedComponents.get_object_sizes(data[1:-1, 1:-1, 1:-1])
        return id_to_volume_dict, touching_ids

    @staticmethod
    def __combine_id_to_volume_dicts(dict1, dict2):
        # make dict1 the larger dict
        if len(dict1) < len(dict2):
            dict1, dict2 = dict2, dict1

        dict1 = dict1.copy()
        for id, volume in dict2.items():
            dict1[id] = dict1.get(id, 0) + volume
        return dict1

    @staticmethod
    def __combine_results(results):
        id_to_volume_dict = {}
        touching_ids = set()
        if type(results) is tuple:
            results = [results]

        for current_id_to_volume_dict, current_touching_ids in results:
            id_to_volume_dict = ConnectedComponents.__combine_id_to_volume_dicts(
                id_to_volume_dict, current_id_to_volume_dict
            )
            touching_ids.update(current_touching_ids)
        return id_to_volume_dict, touching_ids

    def get_connected_component_information(self):
        b = db.from_sequence(
            self.blocks, npartitions=min(len(self.blocks), self.num_workers * 10)
        ).map(self.get_connected_component_information_blockwise)

        with dask_util.start_dask(
            self.num_workers,
            "calculate connected component information",
            logger,
        ):
            with io_util.Timing_Messager(
                "Calculating connected component information", logger
            ):
                self.blockwise_results = b.compute()

    def combine_blockwise_results(self):
        b = db.from_sequence(
            self.blockwise_results,
            npartitions=min(len(self.blockwise_results), self.num_workers * 10),
        ).map(self.__combine_results)

        with dask_util.start_dask(
            self.num_workers, "combine blockwise results", logger
        ):
            with io_util.Timing_Messager("Combining blockwise results", logger):
                bagged_results = b.compute()

        with io_util.Timing_Messager("Combining bagged results", logger):
            self.id_to_volume_dict, self.touching_ids = self.__combine_results(
                bagged_results
            )

    def get_final_connected_components(self):
        with io_util.Timing_Messager("Finding connected components", logger):
            connected_ids = self.get_connected_ids(
                self.id_to_volume_dict.keys(), self.touching_ids
            )

        if self.minimum_volume_voxels:
            with io_util.Timing_Messager("Volume filter connected", logger):
                connected_ids, _ = ConnectedComponents.volume_filter_connected_ids(
                    connected_ids,
                    self.id_to_volume_dict,
                    self.minimum_volume_voxels,
                )
        # sort connected_ids by the minimum id in each connected component
        new_ids = [[i + 1] * len(ids) for i, ids in enumerate(connected_ids)]
        old_ids = connected_ids

        new_ids = list(itertools.chain(*new_ids))
        old_ids = list(itertools.chain(*old_ids))

        if len(new_ids) == 0:
            self.relabeling_dictionary = None
            self.new_dtype = np.uint8
        else:
            self.relabeling_dictionary = dict(zip(old_ids, new_ids))
            self.new_dtype = np.min_scalar_type(max(new_ids))

    def relabel_block(self, block: DaskBlock):
        data = to_ndarray_tensorstore(
            self.tmp_blockwise_ds_tensorstore,
            block.write_roi,
            self.voxel_size,
            self.tmp_blockwise_ds.roi.offset,
        )

        if self.relabeling_dictionary is None:
            new_data = data.astype(self.new_dtype)
        else:
            new_data = np.zeros_like(data, dtype=self.new_dtype)

            # get keys and values as lists from self.relabeling_dictionary
            keys, values = zip(*self.relabeling_dictionary.items())
            replace_values(data, list(keys), list(values), new_data)
        del data
        self.output_ds[block.write_roi] = new_data
        return block.write_roi

    def relabel_dataset(self):
        self.output_ds = create_multiscale_dataset(
            self.output_ds_path,
            dtype=self.new_dtype,
            voxel_size=self.voxel_size,
            total_roi=self.roi,
            write_size=self.tmp_blockwise_ds.chunk_shape
            * self.tmp_blockwise_ds.voxel_size,
        )

        b = db.from_sequence(
            self.blocks,
            npartitions=min(len(self.blocks), self.num_workers * 10),
        ).map(self.relabel_block)

        with dask_util.start_dask(self.num_workers, "relabel dataset", logger):
            with io_util.Timing_Messager("Relabeling dataset", logger):
                b.compute()

    def delete_tmp_dataset(self):
        def delete_block(block, depth):
            block_coords = (block.write_roi.begin - self.tmp_blockwise_ds.roi.begin) / (
                self.tmp_blockwise_ds.chunk_shape * self.tmp_blockwise_ds.voxel_size
            )
            block_coords_string = "/".join(
                [f"{block_coords[i]}" for i in range(depth + 1)]
            )
            os.system(f"rm -rf {self.tmp_blockwise_ds_path}/{block_coords_string}")

        for depth in range(2, -1, -1):
            b = db.from_sequence(
                self.blocks,
                npartitions=min(len(self.blocks), self.num_workers * 10),
            ).map(delete_block, depth)

            with dask_util.start_dask(
                self.num_workers, f"delete blockwise depth: {depth}", logger
            ):
                with io_util.Timing_Messager(
                    f"Deleting blockwise depth:{depth}", logger
                ):
                    b.compute()

        base_path, _ = split_dataset_path(self.tmp_blockwise_ds_path)
        os.system(
            f"rm -rf {base_path}/{get_name_from_path(self.tmp_blockwise_ds_path)}"
        )

    def merge_connected_components_across_blocks(self):
        self.blocks = create_blocks(
            self.roi, self.tmp_blockwise_ds, padding=self.tmp_blockwise_ds.voxel_size
        )
        self.get_connected_component_information()
        self.combine_blockwise_results()
        self.get_final_connected_components()
        self.relabel_dataset()
        self.delete_tmp_dataset()

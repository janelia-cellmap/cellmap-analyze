from funlib.persistence import Array, open_ds
from funlib.geometry import Roi
import numpy as np
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.dask_util import (
    DaskBlock,
    create_blocks,
    guesstimate_npartitions,
)
from cellmap_analyze.util.io_util import (
    get_name_from_path,
    open_ds_tensorstore,
    print_with_datetime,
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
        self.compute_args = {}
        if self.num_workers == 1:
            self.compute_args = {"scheduler": "single-threaded"}

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

    @staticmethod
    def get_connected_component_information_blockwise(
        block, tmp_blockwise_ds_tensorstore, voxel_size, tmp_blockwise_ds, connectivity
    ):
        data = to_ndarray_tensorstore(
            tmp_blockwise_ds_tensorstore,
            block.read_roi,
            voxel_size,
            tmp_blockwise_ds.roi.offset,
        )
        # if roi.shape != self.ds.intersect(roi).shape:
        #     data = self.ds.to_ndarray(roi, fill_value=0)
        # else:
        #     data = self.ds.to_ndarray(roi)

        mask = data.astype(bool)
        mask[2:, 2:, 2:] = False
        touching_ids = ConnectedComponents.get_touching_ids(
            data, mask=mask, connectivity=connectivity
        )

        # get information only from actual block(not including padding)
        id_to_volume_dict = ConnectedComponents.get_object_sizes(data[1:-1, 1:-1, 1:-1])
        block.relabeling_dict = {id: 0 for id in id_to_volume_dict.keys()}
        return [block], id_to_volume_dict, touching_ids

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
        blocks = []
        id_to_volume_dict = {}
        touching_ids = set()
        if type(results) is tuple:
            results = [results]

        for block, current_id_to_volume_dict, current_touching_ids in results:
            blocks += block
            id_to_volume_dict = ConnectedComponents.__combine_id_to_volume_dicts(
                id_to_volume_dict, current_id_to_volume_dict
            )
            touching_ids.update(current_touching_ids)

        return blocks, id_to_volume_dict, touching_ids

    def get_connected_component_information(self):
        b = (
            db.from_sequence(
                self.blocks,
                npartitions=guesstimate_npartitions(self.blocks, self.num_workers),
            )
            .map(
                ConnectedComponents.get_connected_component_information_blockwise,
                self.tmp_blockwise_ds_tensorstore,
                self.voxel_size,
                self.tmp_blockwise_ds,
                self.connectivity,
            )
            .reduction(self.__combine_results, self.__combine_results)
        )

        with dask_util.start_dask(
            self.num_workers,
            "calculate connected component information",
            logger,
        ):
            with io_util.Timing_Messager(
                "Calculating connected component information", logger
            ):
                blocks_with_dict, self.id_to_volume_dict, self.touching_ids = b.compute(
                    **self.compute_args
                )
                for block in blocks_with_dict:
                    self.blocks[block.index] = block

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
        del self.id_to_volume_dict, self.touching_ids
        # sort connected_ids by the minimum id in each connected component
        new_ids = [[i + 1] * len(ids) for i, ids in enumerate(connected_ids)]
        old_ids = connected_ids

        new_ids = list(itertools.chain(*new_ids))
        old_ids = list(itertools.chain(*old_ids))

        if len(new_ids) == 0:
            self.new_dtype = np.uint8
        else:
            self.new_dtype = np.min_scalar_type(max(new_ids))
            relabeling_dict = dict(zip(old_ids, new_ids))
            # update blockwise relabeing dicts
            for block in self.blocks:
                block.relabeling_dict = {
                    id: relabeling_dict.get(id, 0)
                    for id in block.relabeling_dict.keys()
                }

            del relabeling_dict

    @staticmethod
    def relabel_block(
        block: DaskBlock,
        tmp_blockwise_ds_tensorstore,
        voxel_size,
        tmp_blockwise_ds,
        new_dtype,
        output_ds,
    ):
        # print_with_datetime(block.relabeling_dict, logger)
        data = to_ndarray_tensorstore(
            tmp_blockwise_ds_tensorstore,
            block.write_roi,
            voxel_size,
            tmp_blockwise_ds.roi.offset,
        )

        if len(block.relabeling_dict) == 0:
            new_data = data.astype(new_dtype)
        else:
            new_data = np.zeros_like(data, dtype=new_dtype)
            keys, values = zip(*block.relabeling_dict.items())
            replace_values(data, list(keys), list(values), new_data)
        del data
        output_ds[block.write_roi] = new_data

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
            npartitions=guesstimate_npartitions(self.blocks, self.num_workers),
        ).map(
            ConnectedComponents.relabel_block,
            self.tmp_blockwise_ds_tensorstore,
            self.voxel_size,
            self.tmp_blockwise_ds,
            self.new_dtype,
            self.output_ds,
        )

        with dask_util.start_dask(self.num_workers, "relabel dataset", logger):
            with io_util.Timing_Messager("Relabeling dataset", logger):
                b.compute(**self.compute_args)

    @staticmethod
    def delete_chunks(block_coords, tmp_blockwise_ds_path):
        block_coords_string = "/".join([str(c) for c in block_coords])
        delete_name = f"{tmp_blockwise_ds_path}/{block_coords_string}"
        if os.path.exists(delete_name) and (
            os.path.isfile(delete_name) or os.listdir(delete_name) == []
        ):
            os.system(f"rm -rf {delete_name}")

    def delete_tmp_dataset(self):
        for depth in range(3, 0, -1):
            all_block_coords = set([block.coords[:depth] for block in self.blocks])
            b = db.from_sequence(
                all_block_coords,
                npartitions=guesstimate_npartitions(self.blocks, self.num_workers),
            ).map(ConnectedComponents.delete_chunks, self.tmp_blockwise_ds_path)

            with dask_util.start_dask(
                self.num_workers, f"delete blockwise depth: {depth}", logger
            ):
                with io_util.Timing_Messager(
                    f"Deleting blockwise depth: {depth}", logger
                ):
                    b.compute(**self.compute_args)

        base_path, _ = split_dataset_path(self.tmp_blockwise_ds_path)
        os.system(
            f"rm -rf {base_path}/{get_name_from_path(self.tmp_blockwise_ds_path)}"
        )

    def merge_connected_components_across_blocks(self):
        self.blocks = create_blocks(
            self.roi, self.tmp_blockwise_ds, padding=self.tmp_blockwise_ds.voxel_size
        )
        self.get_connected_component_information()
        self.get_final_connected_components()
        self.relabel_dataset()
        self.delete_tmp_dataset()

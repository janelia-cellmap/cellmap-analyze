from collections import defaultdict
import pickle
import types
import numpy as np
from tqdm import tqdm
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.block_util import relabel_block
from cellmap_analyze.util.dask_util import (
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
from cellmap_analyze.util.mask_util import MasksFromConfig
from cellmap_analyze.util.zarr_util import create_multiscale_dataset
import skimage


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ConnectedComponents:
    def __init__(
        self,
        output_path,
        input_path=None,
        intensity_threshold_minimum=-1,
        intensity_threshold_maximum=np.inf,  # exclusive
        mask_config=None,
        connected_components_blockwise_path=None,
        roi=None,
        minimum_volume_nm_3=0,
        maximum_volume_nm_3=np.inf,
        num_workers=10,
        connectivity=2,
        delete_tmp=False,
        invert=False,
        calculating_holes=False,
        fill_holes=False,
    ):
        if input_path and connected_components_blockwise_path:
            raise Exception("Cannot provide both input_path and tmp_blockwise_path")
        if not input_path and not connected_components_blockwise_path:
            raise Exception("Must provide either input_path or tmp_blockwise_path")

        if input_path:
            template_idi = self.input_idi = ImageDataInterface(input_path)
        else:
            template_idi = self.connected_components_blockwise_idi = ImageDataInterface(
                connected_components_blockwise_path
            )

        if roi is None:
            self.roi = template_idi.roi
        else:
            self.roi = roi

        self.calculating_holes = calculating_holes
        self.invert = invert
        self.oob_value = None
        if self.calculating_holes:
            self.invert = True
            self.oob_value = np.prod(self.input_idi.ds.shape) * 10

        self.voxel_size = template_idi.voxel_size

        self.do_full_connected_components = False
        if input_path:
            self.input_path = input_path
            output_ds_name = get_name_from_path(output_path)
            output_ds_basepath = split_dataset_path(output_path)[0]
            self.connected_components_blockwise_path = (
                output_ds_basepath + "/" + output_ds_name + "_blockwise"
            )
            self.intensity_threshold_minimum = intensity_threshold_minimum
            self.intensity_threshold_maximum = intensity_threshold_maximum
            create_multiscale_dataset(
                self.connected_components_blockwise_path,
                dtype=np.uint64,
                voxel_size=self.voxel_size,
                total_roi=self.roi,
                write_size=template_idi.chunk_shape * self.voxel_size,
            )
            self.connected_components_blockwise_idi = ImageDataInterface(
                self.connected_components_blockwise_path + "/s0",
                mode="r+",
                custom_fill_value=self.oob_value,
            )
            self.do_full_connected_components = True
        else:
            self.connected_components_blockwise_path = (
                connected_components_blockwise_path
            )

            self.connected_components_blockwise_idi = ImageDataInterface(
                self.connected_components_blockwise_path
            )
        self.output_path = output_path

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
                connectivity=connectivity,
            )

        self.connectivity = connectivity
        self.invert = invert
        self.delete_tmp = delete_tmp
        self.fill_holes = fill_holes

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

    @staticmethod
    def calculate_block_connected_components(
        block_index,
        input_idi: ImageDataInterface,
        connected_components_blockwise_idi: ImageDataInterface,
        intensity_threshold_minimum=-1,
        intensity_threshold_maximum=np.inf,
        calculating_holes=False,
        oob_value=None,
        invert=None,
        mask: MasksFromConfig = None,
        connectivity=2,
    ):
        if calculating_holes:
            invert = True

        block = create_block_from_index(
            connected_components_blockwise_idi,
            block_index,
        )
        input = input_idi.to_ndarray_ts(block.read_roi)
        if invert:
            thresholded = input == 0
        else:
            thresholded = (input >= intensity_threshold_minimum) & (
                input < intensity_threshold_maximum
            )

        if mask:
            mask_block = mask.process_block(roi=block.read_roi)
            thresholded &= mask_block

        connected_components = skimage.measure.label(
            thresholded, connectivity=connectivity
        )

        global_id_offset = block_index * np.prod(
            block.full_block_size / connected_components_blockwise_idi.voxel_size[0]
        )
        connected_components[connected_components > 0] += global_id_offset

        if calculating_holes and block.read_roi.shape != block.read_roi.intersect(
            input_idi.roi
        ):
            idxs = np.where(input == oob_value)
            if len(idxs) > 0:
                # couldnt do inplace for large uint types because it was converting to floats
                relabeled = connected_components.copy()
                ids_to_set_to_zero = np.unique(connected_components[idxs])
                replace_values(
                    connected_components,
                    list(ids_to_set_to_zero),
                    [oob_value] * len(ids_to_set_to_zero),
                    out_array=relabeled,
                )
                connected_components = relabeled

        connected_components_blockwise_idi.ds[block.write_roi] = connected_components

    def calculate_connected_components_blockwise(self):
        num_blocks = dask_util.get_num_blocks(self.input_idi, roi=self.roi)
        block_indexes = list(range(num_blocks))
        b = db.from_sequence(
            block_indexes,
            npartitions=guesstimate_npartitions(block_indexes, self.num_workers),
        ).map(
            ConnectedComponents.calculate_block_connected_components,
            self.input_idi,
            self.connected_components_blockwise_idi,
            self.intensity_threshold_minimum,
            self.intensity_threshold_maximum,
            self.calculating_holes,
            self.oob_value,
            self.invert,
            self.mask,
            self.connectivity,
        )

        with dask_util.start_dask(
            self.num_workers,
            "calculate connected components",
            logger,
        ):
            with io_util.Timing_Messager("Calculating connected components", logger):
                dask_computer(b, self.num_workers, **self.compute_args)

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
        return defaultdict(int, zip(labels, counts))

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
        connected_ids, id_to_volume_dict, minimum_volume_voxels, maximum_volume_voxels
    ):
        kept_ids = []
        removed_ids = []
        for current_connected_ids in connected_ids:
            volume = sum([id_to_volume_dict[id] for id in current_connected_ids])
            if volume >= minimum_volume_voxels and volume <= maximum_volume_voxels:
                kept_ids.append(current_connected_ids)
            else:
                removed_ids.append(current_connected_ids)
        return kept_ids, removed_ids

    @staticmethod
    def get_connected_component_information_blockwise(
        block_index,
        connected_components_blockwise_idi: ImageDataInterface,
        connectivity,
    ):
        try:
            block = create_block_from_index(
                connected_components_blockwise_idi,
                block_index,
                padding=connected_components_blockwise_idi.voxel_size,
            )
            data = connected_components_blockwise_idi.to_ndarray_ts(
                block.read_roi,
            )

            mask = data.astype(bool)
            mask[2:, 2:, 2:] = False
            touching_ids = ConnectedComponents.get_touching_ids(
                data, mask=mask, connectivity=connectivity
            )

            # get information only from actual block(not including padding)
            id_to_volume_dict = ConnectedComponents.get_object_sizes(
                data[1:-1, 1:-1, 1:-1]
            )
            block.relabeling_dict = {id: 0 for id in id_to_volume_dict.keys()}
        except:
            raise Exception(
                f"Error in get_connected_component_information_blockwise {block_index}, {connected_components_blockwise_idi.voxel_size}"
            )
        return [block], id_to_volume_dict, touching_ids

    @staticmethod
    def combine_id_to_volume_dicts(dict1, dict2):
        # make dict1 the larger dict
        if len(dict1) < len(dict2):
            dict1, dict2 = dict2, dict1

        dict1 = dict1.copy()
        for id, volume in dict2.items():
            dict1[id] = dict1.get(id, 0) + volume
        return dict1

    @staticmethod
    def _combine_results(results):
        if type(results) is tuple:
            results = [results]
        elif isinstance(results, types.GeneratorType):
            results = list(results)

        for idx, (block, current_id_to_volume_dict, current_touching_ids) in enumerate(
            tqdm(results)
        ):
            if idx == 0:
                blocks = block
                id_to_volume_dict = current_id_to_volume_dict
                touching_ids = current_touching_ids
                continue

            blocks += block
            for key, value in current_id_to_volume_dict.items():
                id_to_volume_dict[key] += value
            touching_ids.update(current_touching_ids)

        return blocks, id_to_volume_dict, touching_ids

    def get_connected_component_information(self):
        num_blocks = dask_util.get_num_blocks(self.connected_components_blockwise_idi)
        block_indexes = list(range(num_blocks))
        b = db.from_sequence(
            block_indexes,
            npartitions=guesstimate_npartitions(block_indexes, self.num_workers),
        ).map(
            ConnectedComponents.get_connected_component_information_blockwise,
            self.connected_components_blockwise_idi,
            self.connectivity,
        )

        with dask_util.start_dask(
            self.num_workers,
            "calculate connected component information",
            logger,
        ):
            with io_util.Timing_Messager(
                "Calculating connected component information", logger
            ):
                bagged_results = dask_computer(b, self.num_workers, **self.compute_args)

        # moved this out of dask, seems fast enough without having to daskify
        with io_util.Timing_Messager("Combining bagged results", logger):
            blocks_with_dict, self.id_to_volume_dict, self.touching_ids = (
                ConnectedComponents._combine_results(bagged_results)
            )

        self.blocks = [None] * num_blocks
        for block in blocks_with_dict:
            self.blocks[block.index] = block

    @staticmethod
    def write_out_block_objects(
        path,
        blocks,
        num_local_threads_available,
        local_config,
        num_workers,
        compute_args,
        use_new_temp_dir=False,
    ):
        def write_out_block_object(block, tmp_path, use_new_temp_dir=False):
            block_coords_string = "/".join([str(c) for c in block.coords])
            output_path = f"{tmp_path}/{block_coords_string}.pkl"
            if use_new_temp_dir:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # write relabeling dict to pkl file
            with open(f"{output_path}", "wb") as handle:
                pickle.dump(block, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # NOTE: do it the following way since it should be fast enough with 10 workers and then we don't have to bog down distributed stuff
        with dask_util.start_dask(
            num_workers=num_local_threads_available,
            msg="write out blocks",
            logger=logger,
            config=local_config,
        ):
            with io_util.Timing_Messager("Writing out blocks", logger):
                b = db.from_sequence(
                    blocks,
                    npartitions=guesstimate_npartitions(blocks, num_workers),
                ).map(write_out_block_object, path, use_new_temp_dir)
                dask_computer(b, num_local_threads_available, **compute_args)

    def get_final_connected_components(self):
        with io_util.Timing_Messager("Finding connected components", logger):
            connected_ids = self.get_connected_ids(
                self.id_to_volume_dict.keys(), self.touching_ids
            )

        if self.minimum_volume_voxels > 0 or self.maximum_volume_voxels < np.inf:
            with io_util.Timing_Messager("Volume filter connected", logger):
                connected_ids, _ = ConnectedComponents.volume_filter_connected_ids(
                    connected_ids,
                    self.id_to_volume_dict,
                    self.minimum_volume_voxels,
                    self.maximum_volume_voxels,
                )

        if self.calculating_holes:
            connected_ids = [
                current_connected_ids
                for current_connected_ids in connected_ids
                if self.oob_value not in current_connected_ids
            ]

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
    def relabel_block_from_path(
        block_coords,
        input_idi: ImageDataInterface,
        output_idi: ImageDataInterface,
        block_info_basepath=None,
    ):
        if block_info_basepath is None:
            block_info_basepath = input_idi.path
        # read block from pickle file
        block_coords_string = "/".join([str(c) for c in block_coords])
        with open(f"{block_info_basepath}/{block_coords_string}.pkl", "rb") as handle:
            block = pickle.load(handle)
        relabel_block(block, input_idi, output_idi)

    @staticmethod
    def relabel_dataset(
        original_idi,
        output_path,
        blocks,
        roi,
        dtype,
        num_workers,
        compute_args,
        block_info_basepath=None,
    ):
        create_multiscale_dataset(
            output_path,
            dtype=dtype,
            voxel_size=original_idi.voxel_size,
            total_roi=roi,
            write_size=original_idi.chunk_shape * original_idi.voxel_size,
        )
        output_idi = ImageDataInterface(output_path + "/s0", mode="r+")

        block_coords = [block.coords for block in blocks]
        b = db.from_sequence(
            block_coords,
            npartitions=guesstimate_npartitions(blocks, num_workers),
        ).map(
            ConnectedComponents.relabel_block_from_path,
            original_idi,
            output_idi,
            block_info_basepath,
        )

        with dask_util.start_dask(num_workers, "relabel dataset", logger):
            with io_util.Timing_Messager("Relabeling dataset", logger):
                dask_computer(b, num_workers, **compute_args)

    @staticmethod
    def delete_chunks(block_coords, tmp_blockwise_ds_path):
        block_coords_string = "/".join([str(c) for c in block_coords])
        delete_name = f"{tmp_blockwise_ds_path}/{block_coords_string}"
        if os.path.exists(delete_name) and (
            os.path.isfile(delete_name) or os.listdir(delete_name) == []
        ):
            os.system(f"rm -rf {delete_name}")
            if len(block_coords) == 3:
                # if it is the highest level then we also remove the pkl file
                os.system(f"rm -rf {delete_name}.pkl")

    @staticmethod
    def delete_tmp_dataset(path_to_dataset, blocks, num_workers, compute_args):
        for depth in range(3, 0, -1):
            all_block_coords = set([block.coords[:depth] for block in blocks])
            b = db.from_sequence(
                all_block_coords,
                npartitions=guesstimate_npartitions(blocks, num_workers),
            ).map(
                ConnectedComponents.delete_chunks,
                path_to_dataset,
            )

            with dask_util.start_dask(
                num_workers,
                f"delete blockwise depth: {depth}",
                logger,
            ):
                with io_util.Timing_Messager(
                    f"Deleting blockwise depth: {depth}", logger
                ):
                    dask_computer(b, num_workers, **compute_args)

        basepath, _ = split_dataset_path(path_to_dataset)
        os.system(f"rm -rf {basepath}/{get_name_from_path(path_to_dataset)}")

    def get_connected_components(self):
        self.calculate_connected_components_blockwise()
        self.merge_connected_components_across_blocks()

    def merge_connected_components_across_blocks(self):
        # get blockwise connected component information
        self.get_connected_component_information()
        # get final connected components necessary for relabeling, including volume filtering
        self.get_final_connected_components()
        # write out block information
        ConnectedComponents.write_out_block_objects(
            self.connected_components_blockwise_idi.path,
            self.blocks,
            self.num_local_threads_available,
            self.local_config,
            self.num_workers,
            self.compute_args,
        )
        self.relabel_dataset(
            self.connected_components_blockwise_idi,
            self.output_path,
            self.blocks,
            self.roi,
            self.new_dtype,
            self.num_workers,
            self.compute_args,
        )
        if self.delete_tmp:
            ConnectedComponents.delete_tmp_dataset(
                self.connected_components_blockwise_idi.path,
                self.blocks,
                self.num_workers,
                self.compute_args,
            )

        if self.fill_holes:
            from .fill_holes import FillHoles

            fh = FillHoles(
                input_path=self.output_path + "/s0",
                output_path=self.output_path + "_filled",
                num_workers=self.num_workers,
                roi=self.roi,
                connectivity=self.connectivity,
                delete_tmp=self.delete_tmp,
            )
            fh.fill_holes()

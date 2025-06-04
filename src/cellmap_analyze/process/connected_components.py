import pickle
import numpy as np
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
import fastremap
import os
from cellmap_analyze.util.mask_util import MasksFromConfig
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.zarr_util import create_multiscale_dataset_idi
import cc3d
from collections import Counter

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ConnectedComponents(ComputeConfigMixin):
    def __init__(
        self,
        output_path,
        input_path=None,
        intensity_threshold_minimum=-1,
        intensity_threshold_maximum=np.inf,  # exclusive
        mask_config=None,
        connected_components_blockwise_path=None,
        object_labels_path=None,
        roi=None,
        minimum_volume_nm_3=0,
        maximum_volume_nm_3=np.inf,
        num_workers=10,
        connectivity=2,
        delete_tmp=False,
        invert=False,
        calculating_holes=False,
        fill_holes=False,
        chunk_shape=None,
    ):
        super().__init__(num_workers)
        if input_path and connected_components_blockwise_path:
            raise Exception("Cannot provide both input_path and tmp_blockwise_path")
        if not input_path and not connected_components_blockwise_path:
            raise Exception("Must provide either input_path or tmp_blockwise_path")

        if input_path:
            template_idi = self.input_idi = ImageDataInterface(
                input_path, chunk_shape=chunk_shape
            )
        else:
            template_idi = self.connected_components_blockwise_idi = ImageDataInterface(
                connected_components_blockwise_path, chunk_shape=chunk_shape
            )

        self.object_labels_idi = None
        if object_labels_path:
            self.object_labels_idi = ImageDataInterface(
                object_labels_path, chunk_shape=chunk_shape
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
        output_ds_name = get_name_from_path(output_path)
        output_ds_basepath = split_dataset_path(output_path)[0]
        os.makedirs(output_ds_basepath, exist_ok=True)

        if input_path:
            self.input_path = input_path
            self.intensity_threshold_minimum = intensity_threshold_minimum
            self.intensity_threshold_maximum = intensity_threshold_maximum
            self.connected_components_blockwise_idi = create_multiscale_dataset_idi(
                output_ds_basepath + "/" + output_ds_name + "_blockwise",
                dtype=np.uint64,
                voxel_size=self.voxel_size,
                total_roi=self.roi,
                write_size=template_idi.chunk_shape * self.voxel_size,
                custom_fill_value=self.oob_value,
            )
            self.do_full_connected_components = True
        else:
            self.connected_components_blockwise_idi = ImageDataInterface(
                connected_components_blockwise_path, chunk_shape=chunk_shape
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

        connected_components = cc3d.connected_components(
            thresholded,
            connectivity=6 + 12 * (connectivity >= 2) + 8 * (connectivity >= 3),
            binary_image=True,
            out_dtype=np.uint64,
        )

        global_id_offset = block_index * np.prod(
            block.full_block_size / connected_components_blockwise_idi.voxel_size[0],
            dtype=np.uint64,
        )

        connected_components[connected_components > 0] += global_id_offset

        if calculating_holes and block.read_roi.shape != block.read_roi.intersect(
            input_idi.roi
        ):
            idxs = np.where(input == oob_value)
            if len(idxs) > 0:
                ids_to_set_to_zero = fastremap.unique(connected_components[idxs])
                fastremap.remap(
                    connected_components,
                    dict(
                        zip(
                            list(ids_to_set_to_zero),
                            [oob_value] * len(ids_to_set_to_zero),
                        )
                    ),
                    preserve_missing_labels=True,
                    in_place=True,
                )

        connected_components_blockwise_idi.ds[block.write_roi] = connected_components

    def calculate_connected_components_blockwise(self):
        num_blocks = dask_util.get_num_blocks(self.input_idi, roi=self.roi)
        dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            "calculating connected components",
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

    @staticmethod
    def get_touching_ids(data, mask, connectivity=2, object_labels=None):
        """
        Find all pairs of different labels in `data` whose pixels touch
        (with given connectivity), optionally restricted to those within the same object.

        Parameters
        ----------
        data : ndarray of int
            Label image.
        mask : ndarray of bool
            Which pixels to consider in building the graph.
        connectivity : {1, 2}, optional
            Pixel‐connectivity for the graph (4‐ or 8‐connected in 2D).
        object_labels : ndarray of int, optional
            Additional label/image of same shape.  Only touching‐pairs
            whose object_labels match are kept.

        Returns
        -------
        touching_ids : set of tuple(int, int)
            Each tuple (i, j) with i < j indicates two different `data`-labels that touch.
        """
        # initially from here: https://stackoverflow.com/questions/72452267/finding-identity-of-touching-labels-objects-masks-in-images-using-python
        # build pixel‐adjacency graph
        g, nodes = pixel_graph(data, mask=mask, connectivity=connectivity)

        # extract all neighbor‐pairs
        coo = g.tocoo()
        center_coords = nodes[coo.row]
        neighbor_coords = nodes[coo.col]

        center_vals = data.ravel()[center_coords]
        neighbor_vals = data.ravel()[neighbor_coords]

        # if object_labels given, filter to only same-object contacts
        if object_labels is not None:
            obj_center = object_labels.ravel()[center_coords]
            obj_neighbor = object_labels.ravel()[neighbor_coords]
            same_object = obj_center == obj_neighbor
            center_vals = center_vals[same_object]
            neighbor_vals = neighbor_vals[same_object]

        # remove self‐touches, sort pairs so smaller id comes first
        pairs = np.stack([center_vals, neighbor_vals], axis=1)
        unequal = pairs[:, 0] != pairs[:, 1]
        pairs = pairs[unequal]
        pairs.sort(axis=1)  # row‐wise sort

        # unique set of tuples
        touching_ids = {tuple(p) for p in pairs}
        return touching_ids

    @staticmethod
    def get_object_sizes(data):
        labels, counts = fastremap.unique(data[data > 0], return_counts=True)
        return Counter(dict(zip(labels, counts)))

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
        object_labels_idi=None,
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
            object_labels = None
            if object_labels_idi is not None:
                object_labels = object_labels_idi.to_ndarray_ts(block.read_roi)

            mask = data.astype(bool)
            mask[2:, 2:, 2:] = False
            touching_ids = ConnectedComponents.get_touching_ids(
                data, mask=mask, connectivity=connectivity, object_labels=object_labels
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
    def _merge_tuples(
        acc: tuple[list, Counter, set], res: tuple[list | object, dict, set]
    ) -> tuple[list, Counter, set]:
        """
        acc: (all_blocks_acc, counter_acc, touch_acc)
        res: (blk_or_list, cur_id2vol, cur_touch)

        This version checks for overlap and avoids mutating counter_acc in place.
        """
        if acc is None:
            acc = ([], Counter(), set())

        all_blocks_acc, counter_acc, touch_acc = acc
        blk, cur_id2vol, cur_touch = res

        # 1) Normalize `blk` to a list and concatenate
        if isinstance(blk, list):
            new_blocks = all_blocks_acc + blk
        else:
            new_blocks = all_blocks_acc + [blk]

        # 2) Create a new Counter rather than updating in place
        merged_counter = counter_acc + Counter(cur_id2vol)

        # 3) Union the touching‐IDs sets
        new_touch_set = touch_acc.union(cur_touch)

        return (new_blocks, merged_counter, new_touch_set)

    def get_connected_component_information(self):
        num_blocks = dask_util.get_num_blocks(self.connected_components_blockwise_idi)
        all_blocks, self.id_to_volume_dict, self.touching_ids = (
            dask_util.compute_blockwise_partitions(
                num_blocks,
                self.num_workers,
                self.compute_args,
                logger,
                "calculating connected component information",
                ConnectedComponents.get_connected_component_information_blockwise,
                self.connected_components_blockwise_idi,
                self.connectivity,
                self.object_labels_idi,
                merge_fn=ConnectedComponents._merge_tuples,
                merge_identity=None,
            )
        )
        print(self.id_to_volume_dict)

        # Reconstruct self.blocks by index
        self.blocks = [None] * num_blocks
        for blk in all_blocks:
            self.blocks[blk.index] = blk

    @staticmethod
    def write_out_block_objects(
        path,
        blocks,
        num_local_threads_available,
        local_config,
        compute_args,
        use_new_temp_dir=False,
    ):
        def write_partition(blocks, tmp_path, use_new_temp_dir=False):
            for block in blocks:
                coords = "/".join(str(c) for c in block.coords)
                out = os.path.join(tmp_path, f"{coords}.pkl")
                if use_new_temp_dir:
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "wb") as f:
                    pickle.dump(block, f, protocol=pickle.HIGHEST_PROTOCOL)

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
                    npartitions=guesstimate_npartitions(
                        len(blocks), num_local_threads_available
                    ),
                ).map_partitions(write_partition, path, use_new_temp_dir)
                # now you have only `num_workers` tasks, not millions
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
        block_index,
        input_idi: ImageDataInterface,
        output_idi: ImageDataInterface,
        block_info_basepath=None,
        mask: MasksFromConfig = None,
    ):
        # create block from index
        block_coords = create_block_from_index(input_idi, block_index).coords
        if block_info_basepath is None:
            block_info_basepath = input_idi.path
        # read block from pickle file
        block_coords_string = "/".join([str(c) for c in block_coords])
        with open(f"{block_info_basepath}/{block_coords_string}.pkl", "rb") as handle:
            block = pickle.load(handle)
        relabel_block(block, input_idi, output_idi, mask)

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
        mask=None,
    ):
        output_idi = create_multiscale_dataset_idi(
            output_path,
            dtype=dtype,
            voxel_size=original_idi.voxel_size,
            total_roi=roi,
            write_size=original_idi.chunk_shape * original_idi.voxel_size,
        )

        num_blocks = len(blocks)
        dask_util.compute_blockwise_partitions(
            num_blocks,
            num_workers,
            compute_args,
            logger,
            "relabeling dataset",
            ConnectedComponents.relabel_block_from_path,
            original_idi,
            output_idi,
            block_info_basepath,
            mask=mask,
        )

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
            dask_util.delete_tmp_dataset(
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

import numpy as np
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
)
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.io_util import (
    get_name_from_path,
    get_output_path_from_input_path,
    print_with_datetime,
    split_dataset_path,
)

import logging
import networkx as nx
import itertools
import fastremap
import os
from cellmap_analyze.util.mask_util import MasksFromConfig
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.zarr_util import create_multiscale_dataset_idi
from scipy.ndimage import gaussian_filter
import cc3d
from collections import Counter
import shutil
from cellmap_analyze.util.measure_util import trim_array
from cellmap_analyze.cythonizing.touching import get_touching_ids

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
        gaussian_smoothing_sigma_nm=None,
        mask_config=None,
        connected_components_blockwise_path=None,
        object_labels_path=None,
        deduplicate_ids=False,
        binarize=True,  # usually want to threshold and binarize
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

        self.binarize = binarize
        self.deduplicate_ids = deduplicate_ids
        if deduplicate_ids:
            if self.object_labels_idi is not None:
                raise Exception(
                    "If deduplicate_ids is True, object_labels_path must be empty"
                )
            self.object_labels_idi = self.input_idi
            self.binarize = False

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
        output_path = str(output_path).rstrip("/")
        output_ds_basepath = split_dataset_path(output_path)[0]
        os.makedirs(output_ds_basepath, exist_ok=True)

        if input_path:
            self.input_path = input_path
            self.intensity_threshold_minimum = intensity_threshold_minimum
            self.intensity_threshold_maximum = intensity_threshold_maximum

            # Use helper function to generate blockwise path (handles root datasets correctly)
            blockwise_path = get_output_path_from_input_path(output_path, "_blockwise")

            # For new zarr files (root datasets), also create parent directory
            if not blockwise_path.startswith(output_ds_basepath):
                os.makedirs(os.path.dirname(blockwise_path), exist_ok=True)

            self.connected_components_blockwise_idi = create_multiscale_dataset_idi(
                blockwise_path,
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
        self.gaussian_smoothing_sigma_nm = gaussian_smoothing_sigma_nm
        # Use the stripped version of output_path
        self.output_path = output_path

        self.relabeling_dict_path = f"{self.output_path}_relabeling_dict/"

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
        gaussian_smoothing_sigma_nm=None,
        calculating_holes=False,
        oob_value=None,
        invert=None,
        mask: MasksFromConfig = None,
        connectivity=2,
        binarize=True,
    ):
        if calculating_holes:
            invert = True

        padding_nm = 0
        if gaussian_smoothing_sigma_nm:
            gaussian_smoothing_sigma_voxels = (
                gaussian_smoothing_sigma_nm / input_idi.voxel_size[0]
            )
            truncate = 4.0  # default
            padding_voxels = int(truncate * gaussian_smoothing_sigma_voxels + 0.5)
            padding_nm = padding_voxels * input_idi.voxel_size[0]

        block = create_block_from_index(
            connected_components_blockwise_idi, block_index, padding=padding_nm
        )
        if mask:
            # mask block will always just be the normal size, regardless of smoothing and associated padding
            mask_block = create_block_from_index(
                connected_components_blockwise_idi, block_index
            )
            mask_data = mask.process_block(roi=mask_block.read_roi)

            if not np.any(mask_data):
                connected_components_blockwise_idi.ds[block.write_roi] = 0
                return

        input = input_idi.to_ndarray_ts(block.read_roi)

        if gaussian_smoothing_sigma_nm:
            gaussian_smoothing_sigma_voxels = (
                gaussian_smoothing_sigma_nm / input_idi.voxel_size[0]
            )
            input = gaussian_filter(
                input.astype(np.float32),
                sigma=gaussian_smoothing_sigma_voxels,
                mode="nearest",
            )
            input = trim_array(input, padding_voxels)

        cc3d_connectivity = 6 + 12 * (connectivity >= 2) + 8 * (connectivity >= 3)

        if binarize:
            if invert:
                thresholded = input == 0
            else:
                thresholded = (input >= intensity_threshold_minimum) & (
                    input < intensity_threshold_maximum
                )

            if mask:
                thresholded *= mask_data

            connected_components = cc3d.connected_components(
                thresholded,
                connectivity=cc3d_connectivity,
                binary_image=True,
                out_dtype=np.uint64,
            )
        else:
            if mask:
                input *= mask_data

            connected_components = cc3d.connected_components(
                input,
                connectivity=cc3d_connectivity,
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
            f"calculating blockwise connected components for {self.input_idi.path}",
            ConnectedComponents.calculate_block_connected_components,
            self.input_idi,
            self.connected_components_blockwise_idi,
            self.intensity_threshold_minimum,
            self.intensity_threshold_maximum,
            self.gaussian_smoothing_sigma_nm,
            self.calculating_holes,
            self.oob_value,
            self.invert,
            self.mask,
            self.connectivity,
            self.binarize,
        )

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
                padding_direction="neg_with_edge_pos",
            )
            # need to pad into external space (for holes) which - if we only pad in the negative direction - will not happen on the positive ends

            data = connected_components_blockwise_idi.to_ndarray_ts(
                block.read_roi,
            )
            object_labels = None
            if object_labels_idi is not None:
                object_labels = object_labels_idi.to_ndarray_ts(block.read_roi)

            mask = data.astype(bool)
            mask[2:, 2:, 2:] = False
            # most of the time it doesnt matter about the "original data region" since we
            # are using the same connectivity for blockwise and this. however if we generated blockwise
            # independently and want to ensure that is consistent, then we need this
            original_data_region_mask = np.zeros_like(data, dtype=np.bool)
            if (
                block.read_roi.shape - block.write_roi.shape
            ) == connected_components_blockwise_idi.voxel_size * 2:
                # then we have expanded in the positive direction as well:
                original_data_region_mask[1:-1, 1:-1, 1:-1] = True
            else:
                original_data_region_mask[1:, 1:, 1:] = True

            touching_ids = get_touching_ids(
                data,
                mask=mask,
                connectivity=connectivity,
                original_data_region_mask=original_data_region_mask,
                object_labels=object_labels,
            )
            # get information only from actual block(not including padding)
            actual_padding = np.array(
                block.read_roi.shape / connected_components_blockwise_idi.voxel_size
                - connected_components_blockwise_idi.chunk_shape
            )
            if np.any(actual_padding == 2):
                data = data[1:-1, 1:-1, 1:-1]
            else:
                data = data[1:, 1:, 1:]

            id_to_volume_dict = ConnectedComponents.get_object_sizes(data)
        except Exception as e:
            raise Exception(
                f"Error {e} in get_connected_component_information_blockwise {block_index}, {connected_components_blockwise_idi.voxel_size}"
            )

        if object_labels_idi is not None:
            unique_object_labels = fastremap.unique(object_labels[object_labels > 0])
            return (
                set(unique_object_labels),
                id_to_volume_dict,
                touching_ids,
            )

        return id_to_volume_dict, touching_ids

    @staticmethod
    def _merge_tuples(tuples) -> tuple[Counter, set]:
        # then is deduplicate_ids
        original_ids = set()
        id_to_volume_dict = Counter()
        touching_ids = set()
        for current_tuple in tuples:
            if len(current_tuple) == 3:
                # then is deduplicate_ids
                doing_deduplication = True
                (
                    current_original_ids,
                    current_id_to_volume_dict,
                    current_touching_ids,
                ) = current_tuple
                original_ids.update(current_original_ids)
            else:
                doing_deduplication = False
                current_id_to_volume_dict, current_touching_ids = current_tuple
            id_to_volume_dict.update(current_id_to_volume_dict)  # in-place, C‐loop
            touching_ids.update(current_touching_ids)  # in-place, C‐loop
        if doing_deduplication:
            return (original_ids, id_to_volume_dict, touching_ids)
        else:
            return (id_to_volume_dict, touching_ids)

    def get_connected_component_information(self):
        num_blocks = dask_util.get_num_blocks(self.connected_components_blockwise_idi)
        output_tuple = dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            f"getting blockwise connected component information for {self.connected_components_blockwise_idi.path}",
            ConnectedComponents.get_connected_component_information_blockwise,
            self.connected_components_blockwise_idi,
            self.connectivity,
            self.object_labels_idi,
            merge_info=(
                ConnectedComponents._merge_tuples,
                get_output_path_from_input_path(
                    self.output_path, "_tmp_connected_component_info_to_merge"
                ) + "/",
            ),
        )
        if self.object_labels_idi:
            (
                self.original_ids,
                self.id_to_volume_dict,
                self.touching_ids,
            ) = output_tuple
        else:
            self.id_to_volume_dict, self.touching_ids = output_tuple

    @staticmethod
    def write_memmap_relabeling_dicts(relabeling_dict, output_path):
        os.makedirs(output_path, exist_ok=True)

        if relabeling_dict is None or len(relabeling_dict) == 0:
            # If the relabeling dict is empty, then it will all be zeros so write out empty arrays
            old_sorted = np.array([], dtype=np.uint8)
            new_sorted = np.array([], dtype=np.uint8)

        else:
            # 1) Extract keys and values
            first_key, first_val = next(iter(relabeling_dict.items()))

            all_old = np.fromiter(relabeling_dict.keys(), dtype=type(first_key))
            all_new = np.fromiter(relabeling_dict.values(), dtype=type(first_val))

            # 2) Sort by old keyf
            sort_idx = np.argsort(all_old)
            old_sorted = all_old[sort_idx]
            new_sorted = all_new[sort_idx]

        # 3) Save both arrays to disk in .npy format (which is also memory‐mappable)
        np.save(f"{output_path}/old_sorted.npy", old_sorted)
        np.save(f"{output_path}/new_sorted.npy", new_sorted)

    @staticmethod
    def get_updated_relabeling_dict(query_ids, relabeling_dict_path):
        # 1) Load the sorted old→new arrays as memmaps
        old_sorted_mm = np.load(f"{relabeling_dict_path}/old_sorted.npy", mmap_mode="r")
        new_sorted_mm = np.load(f"{relabeling_dict_path}/new_sorted.npy", mmap_mode="r")

        # 2) Start with all outputs = 0
        out = np.zeros(query_ids.shape, dtype=new_sorted_mm.dtype)

        if len(old_sorted_mm) == 0 and len(new_sorted_mm) == 0:
            return dict(zip(query_ids, out))

        # 3) Compute insertion indices for each query_id
        idx = np.searchsorted(old_sorted_mm, query_ids)

        # 4) Build an “in_bounds” mask
        in_bounds = idx < old_sorted_mm.shape[0]

        # 5) Allocate a boolean array for “matches” (initialize False)
        matches = np.zeros_like(in_bounds, dtype=bool)

        # 6) Only compare old_sorted_mm[idx] vs. query_ids where in_bounds is True
        # this is because if volume filtered, they wont be in the array
        valid_indices = np.nonzero(in_bounds)[0]  # positions where idx is < size
        if valid_indices.size > 0:
            # For those positions:
            sub_idx = idx[valid_indices]  # guaranteed < len(old_sorted_mm)
            sub_q = query_ids[valid_indices]
            matches_sub = old_sorted_mm[sub_idx] == sub_q
            matches[valid_indices] = matches_sub

        # 7) Wherever matches == True, fill out from new_sorted_mm
        good = np.nonzero(matches)[0]
        if good.size > 0:
            out[good] = new_sorted_mm[idx[good]]

        # 8) Return a dict mapping each query_id → (new label or 0)
        return dict(zip(query_ids, out))

    def get_final_connected_components(self):
        self.continue_processing = True
        with io_util.TimingMessager("Finding connected components", logger):
            connected_ids = self.get_connected_ids(
                self.id_to_volume_dict.keys(), self.touching_ids
            )

        if self.deduplicate_ids:
            if len(connected_ids) == len(self.original_ids):
                # No duplicates found, skipping processing
                self.continue_processing = False
                print_with_datetime(
                    f"No duplicate ids found ({len(connected_ids)=},{len(self.original_ids)=}, skipping remaining connected components processing.",
                    logger,
                )
                return
            else:
                print_with_datetime(
                    f"Duplicate ids found ({len(connected_ids)=},{len(self.original_ids)=}.",
                    logger,
                )

        if self.minimum_volume_voxels > 0 or self.maximum_volume_voxels < np.inf:
            with io_util.TimingMessager("Volume filter connected", logger):
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
            relabeling_dict = {}
        else:
            self.new_dtype = np.min_scalar_type(max(new_ids))
            relabeling_dict = dict(zip(old_ids, new_ids))

        ConnectedComponents.write_memmap_relabeling_dicts(
            relabeling_dict, self.relabeling_dict_path
        )

    @staticmethod
    def relabel_block_from_path(
        block_index,
        input_idi: ImageDataInterface,
        output_idi: ImageDataInterface,
        relabeling_dict_path: str,
        mask: MasksFromConfig = None,
    ):
        # create block from index
        block = create_block_from_index(input_idi, block_index)
        if mask:
            mask_block = mask.process_block(roi=block.read_roi)
            if not np.any(mask_block):
                output_idi.ds[block.write_roi] = 0
                return

        # All ids must be accounted for in the relabeling dict
        data = input_idi.to_ndarray_ts(
            block.read_roi,
        )
        ids = fastremap.unique(data[data > 0])
        relabeling_dict = ConnectedComponents.get_updated_relabeling_dict(
            ids, relabeling_dict_path
        )

        if mask:
            data *= mask_block

        if len(relabeling_dict) > 0:
            try:
                fastremap.remap(
                    data, relabeling_dict, preserve_missing_labels=True, in_place=True
                )
            except:
                raise Exception(
                    f"Error in relabel_block {block.read_roi}, {list(relabeling_dict.keys())}, {list(relabeling_dict.values())}"
                )

        output_idi.ds[block.write_roi] = data

    @staticmethod
    def relabel_dataset(
        original_idi,
        output_path,
        roi,
        dtype,
        relabeling_dict_path,
        num_workers,
        compute_args,
        mask=None,
    ):
        output_idi = create_multiscale_dataset_idi(
            output_path,
            dtype=dtype,
            voxel_size=original_idi.voxel_size,
            total_roi=roi,
            write_size=original_idi.chunk_shape * original_idi.voxel_size,
        )

        num_blocks = dask_util.get_num_blocks(original_idi, roi=roi)
        dask_util.compute_blockwise_partitions(
            num_blocks,
            num_workers,
            compute_args,
            logger,
            f"relabeling dataset for {original_idi.path}",
            ConnectedComponents.relabel_block_from_path,
            original_idi,
            output_idi,
            relabeling_dict_path,
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

        if self.continue_processing:
            self.relabel_dataset(
                self.connected_components_blockwise_idi,
                self.output_path,
                self.roi,
                self.new_dtype,
                self.relabeling_dict_path,
                self.num_workers,
                self.compute_args,
                mask=None,
            )

        if self.delete_tmp:
            dask_util.delete_tmp_dir_blockwise(
                self.connected_components_blockwise_idi,
                self.num_workers,
                self.compute_args,
            )

        if not self.continue_processing:
            return

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

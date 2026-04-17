# %%
from typing import Union, List
from cellmap_analyze.util import io_util
from cellmap_analyze.util.io_util import (
    get_leaf_name_from_path,
    get_output_path_from_input_path,
)
from cellmap_analyze.util.image_data_interface import (
    ImageDataInterface,
)
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util import dask_util
from cellmap_analyze.util.dask_util import create_block_from_index
from funlib.geometry import Coordinate
import logging
import pandas as pd
import numpy as np
import os
from scipy import spatial
import fastremap
import fastmorph

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AssignToOrganelles(ComputeConfigMixin):
    def __init__(
        self,
        organelle_csvs: Union[str, List[str]],
        target_organelle_ds_path: str,
        output_path: str,
        assignment_type: int = 0,
        iteration_distance_nm=10_000,
        num_workers: int = 1,
    ):
        super().__init__(num_workers)
        if isinstance(organelle_csvs, str):
            organelle_csvs = [organelle_csvs]

        self.organelle_info_dict = {}

        for organelle_csv in organelle_csvs:
            df = pd.read_csv(organelle_csv)
            if "Total Objects" in df.columns:
                # delete last two columns of dataframe
                df = df.iloc[:, :-2]
            self.organelle_info_dict[organelle_csv] = df

        self.organelle_idi = ImageDataInterface(target_organelle_ds_path)
        self.organelle_name = get_leaf_name_from_path(
            target_organelle_ds_path
        ).capitalize()
        self.assignment_type = assignment_type
        self.output_path = str(output_path).rstrip("/")
        self.iteration_distance_nm = iteration_distance_nm

    @staticmethod
    def _group_coms_by_block(coms_nm, organelle_idi, com_row_indices=None):
        """Group COM positions by block index.

        Args:
            coms_nm: Array of COM positions in nm, shape (N, 3).
            organelle_idi: ImageDataInterface for the organelle dataset.
            com_row_indices: Optional list of original DataFrame row indices.
                If None, uses range(len(coms_nm)).

        Returns:
            Tuple of (block_to_com_rows dict, nchunks array).
            block_to_com_rows maps block_linear_index -> list of df row indices.
        """
        if com_row_indices is None:
            com_row_indices = list(range(len(coms_nm)))

        voxel_size = np.array(organelle_idi.voxel_size)
        chunk_shape = np.array(organelle_idi.chunk_shape)
        if len(chunk_shape) == 4:
            chunk_shape = chunk_shape[1:]
        block_size = chunk_shape * voxel_size
        roi_start = np.array(organelle_idi.roi.get_begin())
        roi_end = np.array(organelle_idi.roi.get_end())
        nchunks = np.ceil((roi_end - roi_start) / block_size).astype(int)

        # Compute block coordinates for each COM, clamp to valid range
        block_coords = ((coms_nm - roi_start) / block_size).astype(int)
        block_coords = np.clip(block_coords, 0, nchunks - 1)

        # Convert to linear indices
        block_indices = np.ravel_multi_index(block_coords.T, nchunks)

        # Group by block
        block_to_com_rows = {}
        for i, block_idx in enumerate(block_indices):
            block_to_com_rows.setdefault(int(block_idx), []).append(
                com_row_indices[i]
            )

        return block_to_com_rows, nchunks

    @staticmethod
    def _process_containing_block(
        partition_index,
        organelle_idi,
        coms_path,
        block_indices,
        block_to_com_rows,
    ):
        block_index = block_indices[partition_index]
        com_rows = block_to_com_rows[block_index]

        coms_nm = np.load(coms_path, mmap_mode="r")

        block = create_block_from_index(organelle_idi, block_index)
        seg = organelle_idi.to_ndarray_ts(block.read_roi)

        voxel_size = np.array(organelle_idi.voxel_size)
        roi_begin = np.array(block.read_roi.get_begin())

        results = {}
        for row_idx in com_rows:
            com = coms_nm[row_idx]
            local_idx = ((com - roi_begin) / voxel_size).astype(int)

            if np.all(local_idx >= 0) and np.all(local_idx < seg.shape):
                results[row_idx] = int(
                    seg[local_idx[0], local_idx[1], local_idx[2]]
                )
            else:
                results[row_idx] = 0

        return results

    @staticmethod
    def _process_n_nearest_block(
        partition_index,
        organelle_idi,
        coms_path,
        block_indices,
        block_to_com_rows,
        n,
        initial_padding,
    ):
        block_index = block_indices[partition_index]
        com_rows = block_to_com_rows[block_index]

        coms_nm = np.load(coms_path, mmap_mode="r")

        voxel_size = np.array(organelle_idi.voxel_size)
        padding = initial_padding
        dataset_roi = organelle_idi.roi
        results = {}

        while True:
            block = create_block_from_index(
                organelle_idi,
                block_index,
                padding=padding,
                read_beyond_roi=False,
            )
            # Check if we've covered the full dataset
            covers_full_dataset = block.read_roi.contains(dataset_roi)
            seg = organelle_idi.to_ndarray_ts(block.read_roi)

            roi_begin = np.array(block.read_roi.get_begin())
            roi_end = np.array(block.read_roi.get_end())

            # Extract boundaries
            boundaries = seg - fastmorph.erode(seg, erode_border=False)
            boundary_voxels = np.argwhere(boundaries > 0)

            if len(boundary_voxels) == 0:
                padding = Coordinate(p * 2 for p in padding)
                continue

            boundary_ids = boundaries[
                boundary_voxels[:, 0],
                boundary_voxels[:, 1],
                boundary_voxels[:, 2],
            ]
            boundary_coords_nm = (
                roi_begin + (boundary_voxels + 0.5) * voxel_size
            )
            unique_boundary_ids = fastremap.unique(boundary_ids)

            # Filter to only COMs not yet verified
            pending_rows = [r for r in com_rows if r not in results]
            if not pending_rows:
                break

            pending_coms = np.array([coms_nm[r] for r in pending_rows])

            # Margins: min distance from each COM to any face of read_roi
            # that is interior to the dataset (faces at the dataset edge
            # don't need checking — there's nothing beyond them)
            ds_begin = np.array(dataset_roi.get_begin())
            ds_end = np.array(dataset_roi.get_end())
            face_dists_neg = pending_coms - roi_begin  # (C, 3)
            face_dists_pos = roi_end - pending_coms  # (C, 3)
            # Mask out faces that touch the dataset boundary (set to inf)
            at_ds_begin = np.isclose(roi_begin, ds_begin)
            at_ds_end = np.isclose(roi_end, ds_end)
            face_dists_neg[:, at_ds_begin] = np.inf
            face_dists_pos[:, at_ds_end] = np.inf
            margins = np.minimum(
                np.min(face_dists_neg, axis=1),
                np.min(face_dists_pos, axis=1),
            )

            # Containing organelle check (vectorized)
            local_inds = ((pending_coms - roi_begin) / voxel_size).astype(
                int
            )
            in_bounds = np.all(local_inds >= 0, axis=1) & np.all(
                local_inds < seg.shape, axis=1
            )
            containing_ids = np.zeros(len(pending_rows), dtype=int)
            if np.any(in_bounds):
                valid = local_inds[in_bounds]
                containing_ids[in_bounds] = seg[
                    valid[:, 0], valid[:, 1], valid[:, 2]
                ]

            # Per-organelle KDTree: build one tree per organelle,
            # query all COMs for nearest boundary point
            num_unique = len(unique_boundary_ids)
            min_dist_per_org = np.full(
                (len(pending_rows), num_unique), np.inf
            )
            for j, uid in enumerate(unique_boundary_ids):
                mask = boundary_ids == uid
                tree = spatial.KDTree(boundary_coords_nm[mask])
                dists, _ = tree.query(pending_coms)
                min_dist_per_org[:, j] = dists

                # Override distance to 0 for COMs inside this organelle
                is_containing = containing_ids == uid
                if np.any(is_containing):
                    min_dist_per_org[is_containing, j] = 0.0

            # For each COM, sort organelles by distance, take top n
            sort_order = np.argsort(min_dist_per_org, axis=1)
            sorted_dists = np.take_along_axis(
                min_dist_per_org, sort_order, axis=1
            )
            sorted_ids = unique_boundary_ids[sort_order]

            unverified_rows = []
            for i, row_idx in enumerate(pending_rows):
                top_n_ids = np.zeros(n, dtype=int)
                top_n_dists = np.full(n, np.inf)
                num_to_take = min(n, num_unique)
                top_n_ids[:num_to_take] = sorted_ids[i, :num_to_take]
                top_n_dists[:num_to_take] = sorted_dists[i, :num_to_take]

                # Handle containing organelle not in boundary set
                cid = containing_ids[i]
                if cid > 0 and cid not in top_n_ids[:num_to_take]:
                    top_n_ids[num_to_take - 1] = cid
                    top_n_dists[num_to_take - 1] = 0.0
                    order = np.argsort(top_n_dists)
                    top_n_ids = top_n_ids[order]
                    top_n_dists = top_n_dists[order]

                d_n = top_n_dists[n - 1] if num_unique >= n else np.inf
                if covers_full_dataset or d_n < margins[i]:
                    results[row_idx] = {
                        "ids": top_n_ids,
                        "distances": top_n_dists,
                    }
                else:
                    unverified_rows.append(row_idx)

            if not unverified_rows:
                break

            # Double padding and retry for unverified COMs only
            padding = Coordinate(p * 2 for p in padding)
            com_rows = unverified_rows

        return results

    @staticmethod
    def _merge_dicts(list_of_dicts):
        merged = {}
        for d in list_of_dicts:
            if d is not None:
                merged.update(d)
        return merged

    def _save_coms_to_tmp(self, coms):
        """Save COM array to a temp file for workers to read."""
        tmp_dir = os.path.join(self.output_path, ".tmp_assign")
        os.makedirs(tmp_dir, exist_ok=True)
        coms_path = os.path.join(tmp_dir, "coms.npy")
        np.save(coms_path, coms)
        return coms_path

    def assign_to_containing_organelle(self, df, organelle_name):
        coms = df[["COM Z (nm)", "COM Y (nm)", "COM X (nm)"]].to_numpy()
        sf = self.organelle_idi.voxel_size_scale_factor
        coms_scaled = coms * sf
        coms_path = self._save_coms_to_tmp(coms_scaled)

        block_to_com_rows, _ = self._group_coms_by_block(
            coms_scaled, self.organelle_idi
        )
        block_indices = sorted(block_to_com_rows.keys())

        if not block_indices:
            return

        output_dir = os.path.join(self.output_path, ".tmp_assign_containing")
        results = dask_util.compute_blockwise_partitions(
            len(block_indices),
            self.num_workers,
            self.compute_args,
            logger,
            f"assigning containing {organelle_name}",
            AssignToOrganelles._process_containing_block,
            self.organelle_idi,
            coms_path,
            block_indices,
            block_to_com_rows,
            merge_info=(AssignToOrganelles._merge_dicts, output_dir),
        )

        id_col = f"{organelle_name} ID"
        for row_idx, org_id in results.items():
            df.at[row_idx, id_col] = org_id
        df[id_col] = df[id_col].astype(int)

        os.remove(coms_path)

    def assign_to_n_nearest_organelles(self, df, n, organelle_name):
        coms = df[["COM Z (nm)", "COM Y (nm)", "COM X (nm)"]].to_numpy()
        sf = self.organelle_idi.voxel_size_scale_factor
        coms_scaled = coms * sf
        coms_path = self._save_coms_to_tmp(coms_scaled)

        voxel_size = np.array(self.organelle_idi.voxel_size)
        chunk_shape = np.array(self.organelle_idi.chunk_shape)
        if len(chunk_shape) == 4:
            chunk_shape = chunk_shape[1:]
        block_size_nm = chunk_shape * voxel_size

        # Initial padding: half block size, aligned to voxel grid
        half_block = max(block_size_nm) / 2
        padding = Coordinate(
            int(np.ceil(half_block / vs)) * vs for vs in voxel_size
        )

        block_to_com_rows, _ = self._group_coms_by_block(
            coms_scaled, self.organelle_idi
        )
        block_indices = sorted(block_to_com_rows.keys())

        if not block_indices:
            return

        output_dir = os.path.join(self.output_path, ".tmp_assign_nearest")
        # Each block handles its own padding expansion internally
        results = dask_util.compute_blockwise_partitions(
            len(block_indices),
            self.num_workers,
            self.compute_args,
            logger,
            f"assigning {n} nearest {organelle_name}",
            AssignToOrganelles._process_n_nearest_block,
            self.organelle_idi,
            coms_path,
            block_indices,
            block_to_com_rows,
            n,
            padding,
            merge_info=(AssignToOrganelles._merge_dicts, output_dir),
        )

        os.remove(coms_path)

        # Apply results to DataFrame
        id_col = f"{organelle_name} ID"
        dist_col = f"{organelle_name} Distance (nm)"

        if n > 1:
            df[id_col] = [[] for _ in range(len(df))]
            df[dist_col] = [[] for _ in range(len(df))]
            for row_idx, result in results.items():
                df.at[row_idx, id_col] = result["ids"].tolist()
                df.at[row_idx, dist_col] = (result["distances"] / sf).tolist()
        else:
            for row_idx, result in results.items():
                df.at[row_idx, id_col] = int(result["ids"][0])
                df.at[row_idx, dist_col] = float(result["distances"][0] / sf)

    def assign_to_organelles(self):
        with io_util.TimingMessager("Assigning objects to organelles", logger):
            for organelle_csv, df in self.organelle_info_dict.items():
                id_col = f"{self.organelle_name} ID"
                df[id_col] = 0
                if self.assignment_type == 0:
                    self.assign_to_containing_organelle(
                        df, self.organelle_name
                    )
                    continue
                dist_col = f"{self.organelle_name} Distance (nm)"
                df[dist_col] = 0
                self.assign_to_n_nearest_organelles(
                    df, self.assignment_type, self.organelle_name
                )

    def write_updated_csvs(self):
        name = self.organelle_name.lower()
        with io_util.TimingMessager("Writing out updated dataframes", logger):
            os.makedirs(self.output_path, exist_ok=True)
            for csv, df in self.organelle_info_dict.items():
                csv_name = os.path.basename(csv.split(".csv")[0])
                output_path = self.output_path
                if csv_name.endswith("contacts"):  # pragma: no cover
                    # Use helper function to generate contact_sites path (handles root datasets correctly)
                    output_path = get_output_path_from_input_path(
                        self.output_path, "/contact_sites"
                    )
                    os.makedirs(output_path, exist_ok=True)

                if self.assignment_type == 0:
                    output_name = (
                        f"{output_path}/{csv_name}_assigned_to_containing_{name}"
                    )
                elif self.assignment_type == 1:
                    output_name = (
                        f"{output_path}/{csv_name}_assigned_to_nearest_{name}"
                    )
                else:
                    output_name = f"{output_path}/{csv_name}_assigned_to_{self.assignment_type}_nearest_{name}s"
                df["Object ID"] = df["Object ID"].astype(
                    int
                )  # in case was converted to float
                df.to_csv(output_name + ".csv", index=False)

    def get_organelle_assignments(self):
        self.assign_to_organelles()
        self.write_updated_csvs()

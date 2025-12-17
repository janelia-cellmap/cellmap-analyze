# %%
from typing import Union, List
from cellmap_analyze.util import io_util
from cellmap_analyze.util.io_util import get_output_path_from_input_path
from cellmap_analyze.util.image_data_interface import (
    ImageDataInterface,
)
import logging
import pandas as pd
import numpy as np
import os
from scipy import spatial
from tqdm import tqdm
import fastremap
import fastmorph

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AssignToCells:
    def __init__(
        self,
        organelle_csvs: Union[str, List[str]],
        cell_ds_path: str,
        output_path: str,
        cell_assignment_type: int = 0,
        iteration_distance_nm=10_000,
    ):
        if isinstance(organelle_csvs, str):
            organelle_csvs = [organelle_csvs]

        self.organelle_info_dict = {}

        for organelle_csv in organelle_csvs:
            df = pd.read_csv(organelle_csv)
            if "Total Objects" in df.columns:
                # delete last two columns of dataframe
                df = df.iloc[:, :-2]
            self.organelle_info_dict[organelle_csv] = df

        self.cell_idi = ImageDataInterface(cell_ds_path)
        self.cell_assignment_type = cell_assignment_type
        self.output_path = str(output_path).rstrip("/")
        self.iteration_distance_nm = iteration_distance_nm

    @staticmethod
    def assign_to_containing_cell(cell_idi, df):
        cell_data = cell_idi.to_ndarray_ts()
        coms = df[["COM Z (nm)", "COM Y (nm)", "COM X (nm)"]].to_numpy()
        # Already have top left corners aligned since in measure_util get_region_properties we already center on voxel so that top left corners are aligned
        inds = np.astype(coms // cell_idi.voxel_size, int)
        in_bounds = np.all((inds >= cell_idi.domain.inclusive_min), axis=1) & np.all(
            (inds < cell_idi.domain.exclusive_max), axis=1
        )

        df.loc[in_bounds, "Cell ID"] = cell_data[
            inds[in_bounds, 0], inds[in_bounds, 1], inds[in_bounds, 2]
        ]
        df["Cell ID"] = df["Cell ID"].astype(int)

    @staticmethod
    def assign_to_n_nearest_cells(cell_idi, df, n, iteration_distance_nm):
        coms = df[["COM Z (nm)", "COM Y (nm)", "COM X (nm)"]].to_numpy()
        inds = np.astype(coms // cell_idi.voxel_size, int)
        cell_data = cell_idi.to_ndarray_ts()
        if len(fastremap.unique(cell_data[cell_data > 0])) < n:
            raise ValueError(
                f"Number of unique cell ids in the segmentation ({fastremap.unique(cell_data[cell_data > 0])}) is less than n ({n})."
                " Please choose a smaller n or use a different assignment method."
            )

        num_points = len(coms)
        boundaries = cell_data - fastmorph.erode(cell_data, erode_border=False)
        unique_ids = fastremap.unique(boundaries[boundaries > 0])

        # Initialize arrays to store the n closest distances and corresponding ids.
        # Every query point gets an array of size n.
        closest_distances = np.full((num_points, n), np.inf)
        closest_ids = np.zeros((num_points, n), dtype=int)

        boundary_coords = np.argwhere(boundaries)
        boundary_ids = boundaries[
            boundary_coords[:, 0], boundary_coords[:, 1], boundary_coords[:, 2]
        ]

        maximum_distance = iteration_distance_nm
        iteration = 0

        # Continue looping until every query point has n finite (non-inf) distances.
        while np.any(np.any(np.isinf(closest_distances), axis=1)):
            iteration += 1
            # Global update mask: select only those query points that still have at least one np.inf.
            global_update_mask = np.any(np.isinf(closest_distances), axis=1)
            if not np.any(global_update_mask):
                break

            # Loop over each unique boundary id.
            for unique_id in tqdm(unique_ids):
                # Create an update mask for query points that need updates and do not already have this unique_id.
                # We check along the row: if the current candidate list already contains this unique_id, skip updating it.
                already_has_id = np.any(closest_ids == unique_id, axis=1)
                update_mask = global_update_mask & ~already_has_id
                if not np.any(update_mask):
                    continue

                # For the current unique object, get its boundary voxel coordinates, adjust by 0.5 for centering and scale.
                coords = (
                    boundary_coords[boundary_ids == unique_id] + 0.5
                ) * cell_idi.voxel_size
                tree = spatial.KDTree(coords)

                # Query only for the points (coms) that need updating.
                current_distances, _ = tree.query(
                    coms[update_mask],
                    distance_upper_bound=maximum_distance * iteration,
                )
                # Check if coms are within a cell
                updated_inds = inds[update_mask]
                in_bounds = np.all(
                    (updated_inds >= cell_idi.domain.inclusive_min), axis=1
                ) & np.all((updated_inds < cell_idi.domain.exclusive_max), axis=1)
                valid_inds = updated_inds[in_bounds]

                # Initialize an array of False of the same length as updated_inds.
                within_cell = np.full(updated_inds.shape[0], False, dtype=bool)

                # For the indices that are in bounds, assign the comparison result.
                within_cell[in_bounds] = (
                    cell_data[valid_inds[:, 0], valid_inds[:, 1], valid_inds[:, 2]]
                    == unique_id
                )

                # If the com is within a cell, set the distance to 0
                current_distances[within_cell] = 0

                # Combine the current n best distances with the new candidate (this gives n+1 candidates per query point).
                combined_distances = np.column_stack(
                    [closest_distances[update_mask], current_distances]
                )
                combined_ids = np.column_stack(
                    [
                        closest_ids[update_mask],
                        np.full(np.sum(update_mask), unique_id, dtype=int),
                    ]
                )

                # For each query point, sort candidates so that the smallest distances come first.
                sort_order = np.argsort(combined_distances, axis=1)
                sorted_distances = np.take_along_axis(
                    combined_distances, sort_order, axis=1
                )
                sorted_ids = np.take_along_axis(combined_ids, sort_order, axis=1)

                # Update the candidate arrays for only the query points that needed an update.
                closest_distances[update_mask] = sorted_distances[:, :n]
                closest_ids[update_mask] = sorted_ids[:, :n]

        # Update the DataFrame columns.
        if n > 1:
            df["Cell ID"] = [row.tolist() for row in closest_ids]
            df["Cell Distance (nm)"] = [row.tolist() for row in closest_distances]
        else:
            df["Cell ID"] = closest_ids[:, 0]
            df["Cell Distance (nm)"] = closest_distances[:, 0]

    def assign_to_cells(self):
        with io_util.TimingMessager("Assigning objects to cells", logger):
            for organelle_csv, df in self.organelle_info_dict.items():
                # get filename from organelle_csv
                filename = os.path.basename(organelle_csv)
                if filename == "cell.csv":
                    continue
                df["Cell ID"] = 0
                if "er.csv" in organelle_csv:
                    df["Cell ID"] = df["Object ID"]
                    continue
                if self.cell_assignment_type == 0:
                    self.assign_to_containing_cell(self.cell_idi, df)
                    continue
                df["Cell Distance (nm)"] = 0
                self.assign_to_n_nearest_cells(
                    self.cell_idi,
                    df,
                    self.cell_assignment_type,
                    self.iteration_distance_nm,
                )

    def write_updated_csvs(self):
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

                if self.cell_assignment_type == 0:
                    output_name = (
                        f"{output_path}/{csv_name}_assigned_to_containing_cell"
                    )
                elif self.cell_assignment_type == 1:
                    output_name = f"{output_path}/{csv_name}_assigned_to_nearest_cell"
                else:
                    output_name = f"{output_path}/{csv_name}_assigned_to_{self.cell_assignment_type}_nearest_cells"
                df["Object ID"] = df["Object ID"].astype(
                    int
                )  # in case was converted to float
                df.to_csv(output_name + ".csv", index=False)

    def get_cell_assignments(self):
        self.assign_to_cells()
        self.write_updated_csvs()

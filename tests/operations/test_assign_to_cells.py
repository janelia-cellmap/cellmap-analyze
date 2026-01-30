# %%
import pytest
import cellmap_analyze.analyze.assign_to_cells
from importlib import reload

reload(cellmap_analyze.analyze.assign_to_cells)

from cellmap_analyze.analyze.assign_to_cells import AssignToCells
from cellmap_analyze.util.xarray_image_data_interface import (
    XarrayImageDataInterface,
)
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from pandas.testing import assert_frame_equal


def get_id_if_in_cell(coord, cell_data):
    """
    Given a coordinate, check if it is within the bounds of the cell_data array.
    If it is, return the cell ID at that coordinate; otherwise, return 0.
    """
    if np.all(coord >= 0) and np.all(coord < cell_data.shape):
        return cell_data[tuple(coord)]
    else:
        return 0


def gt_assignments(organelle_csv, cell_ds_path, n):
    # Instantiate your image data interface and get the 3D array.
    cell_idi = XarrayImageDataInterface(cell_ds_path)
    cell_data = (
        cell_idi.to_ndarray_ts()
    )  # 3D numpy array containing cell IDs (nonzero entries)

    # Read the COMs from the CSV file.
    # The CSV is expected to have columns: "Object ID", "COM Z (nm)", "COM Y (nm)", "COM X (nm)"
    df_com = pd.read_csv(organelle_csv)
    # Extract COM coordinates in (Z, Y, X) order to match cell_coords.
    coms = df_com[["COM Z (nm)", "COM Y (nm)", "COM X (nm)"]].to_numpy()
    com_coords = np.array(coms // cell_idi.voxel_size, dtype=int)

    # Prepare a dictionary to store the assignments for each object.
    cell_assignment_ids = []
    cell_assignment_distances = []
    if n == 0:
        for i in range(len(df_com)):
            current_com_coords = com_coords[i]
            current_cell_id = get_id_if_in_cell(current_com_coords, cell_data)
            # Save the assignment for this object.

            cell_assignment_ids.append(current_cell_id)
        df_com["Cell ID"] = cell_assignment_ids
        return df_com

    # Get the indices of all nonzero elements in cell_data.
    # Each row in cell_indices is an index triplet (Z, Y, X)
    cell_indices = np.argwhere(cell_data)

    # Compute the physical coordinates by scaling indices with voxel size.
    # cell_coords will have the same shape as cell_indices.
    cell_coords = (cell_indices + 0.5) * cell_idi.voxel_size

    # Extract the corresponding cell IDs from the cell_data array.
    # cell_ids is a 1D array that, for each nonzero coordinate in cell_coords, gives its cell ID.
    cell_ids = cell_data[tuple(cell_indices.T)]

    # Compute pairwise Euclidean distances between each COM and each cell coordinate.
    # distance_matrix has shape (n_objects, n_cells)
    distance_matrix = cdist(coms, cell_coords, metric="euclidean")

    # Loop over each object from the CSV.
    for i in range(len(df_com)):
        # Get the vector of distances from the current COM to all cell coordinates.
        distances = distance_matrix[i, :]
        surrounding_cell_id = get_id_if_in_cell(com_coords[i], cell_data)
        if surrounding_cell_id != 0:
            # If the COM is within a cell, set the distance to zero for that cell ID.
            distances[cell_ids == surrounding_cell_id] = 0
        # Find the indices of the n smallest distances.

        nearest_indices = np.argsort(distances)
        nearest_distances = []
        nearest_cell_ids = []

        for nearest_index in nearest_indices:
            current_cell_id = cell_ids[nearest_index]
            if current_cell_id not in nearest_cell_ids and len(nearest_cell_ids) < n:
                nearest_cell_ids.append(current_cell_id)
                nearest_distances.append(distances[nearest_index])
            if len(nearest_cell_ids) == n:
                cell_assignment_ids.append(
                    nearest_cell_ids
                )  # The cell IDs from the cell_data array
                cell_assignment_distances.append(
                    nearest_distances
                )  # Their corresponding distances to the COM
                break

            # Save the assignment for this object.
    if n > 1:
        df_com["Cell ID"] = cell_assignment_ids
        df_com["Cell Distance (nm)"] = cell_assignment_distances
    else:
        df_com["Cell ID"] = [current_ids[0] for current_ids in cell_assignment_ids]
        df_com["Cell Distance (nm)"] = [
            current_distances[0] for current_distances in cell_assignment_distances
        ]
    return df_com


@pytest.mark.parametrize("cell_assignment_type", [0, 1, 2, 3])
def test_assign_to_cells(
    shared_tmpdir,
    tmp_zarr,
    cell_assignment_type,
):

    atc = AssignToCells(
        organelle_csvs=f"{shared_tmpdir}/csvs/assignment_coms.csv",
        cell_ds_path=f"{tmp_zarr}/segmentation_cells/s0",
        output_path=f"{shared_tmpdir}/csvs/",
        cell_assignment_type=cell_assignment_type,
    )
    atc.get_cell_assignments()
    test_df = list(atc.organelle_info_dict.values())[0]
    gt_df = gt_assignments(
        f"{shared_tmpdir}/csvs/assignment_coms.csv",
        f"{tmp_zarr}/segmentation_cells/s0",
        cell_assignment_type,
    )
    assert_frame_equal(gt_df, test_df, check_dtype=False)

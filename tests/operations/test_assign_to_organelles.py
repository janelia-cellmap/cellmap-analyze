# %%
import pytest
import cellmap_analyze.analyze.assign_to_organelles
from importlib import reload

reload(cellmap_analyze.analyze.assign_to_organelles)

from cellmap_analyze.analyze.assign_to_organelles import AssignToOrganelles
from cellmap_analyze.util.image_data_interface import (
    ImageDataInterface,
)
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from pandas.testing import assert_frame_equal


def get_id_if_in_organelle(coord, organelle_data):
    """
    Given a coordinate, check if it is within the bounds of the organelle_data array.
    If it is, return the organelle ID at that coordinate; otherwise, return 0.
    """
    if np.all(coord >= 0) and np.all(coord < organelle_data.shape):
        return organelle_data[tuple(coord)]
    else:
        return 0


def gt_assignments(organelle_csv, target_organelle_ds_path, n):
    # Instantiate your image data interface and get the 3D array.
    organelle_idi = ImageDataInterface(target_organelle_ds_path)
    organelle_data = organelle_idi.to_ndarray_ts()

    # Read the COMs from the CSV file.
    # The CSV is expected to have columns: "Object ID", "COM Z (nm)", "COM Y (nm)", "COM X (nm)"
    df_com = pd.read_csv(organelle_csv)
    # Extract COM coordinates in (Z, Y, X) order to match organelle_coords.
    coms = df_com[["COM Z (nm)", "COM Y (nm)", "COM X (nm)"]].to_numpy()
    com_coords = np.array(coms // organelle_idi.voxel_size, dtype=int)

    # Prepare lists to store the assignments for each object.
    assignment_ids = []
    assignment_distances = []
    if n == 0:
        for i in range(len(df_com)):
            current_com_coords = com_coords[i]
            current_id = get_id_if_in_organelle(current_com_coords, organelle_data)
            assignment_ids.append(current_id)
        df_com["Segmentation_cells ID"] = assignment_ids
        return df_com

    # Get the indices of all nonzero elements in organelle_data.
    organelle_indices = np.argwhere(organelle_data)

    # Compute the physical coordinates by scaling indices with voxel size.
    organelle_coords = (organelle_indices + 0.5) * organelle_idi.voxel_size

    # Extract the corresponding organelle IDs from the organelle_data array.
    organelle_ids = organelle_data[tuple(organelle_indices.T)]

    # Compute pairwise Euclidean distances between each COM and each organelle coordinate.
    distance_matrix = cdist(coms, organelle_coords, metric="euclidean")

    # Loop over each object from the CSV.
    for i in range(len(df_com)):
        distances = distance_matrix[i, :]
        surrounding_id = get_id_if_in_organelle(com_coords[i], organelle_data)
        if surrounding_id != 0:
            distances[organelle_ids == surrounding_id] = 0

        nearest_indices = np.argsort(distances)
        nearest_distances = []
        nearest_ids = []

        for nearest_index in nearest_indices:
            current_id = organelle_ids[nearest_index]
            if current_id not in nearest_ids and len(nearest_ids) < n:
                nearest_ids.append(current_id)
                nearest_distances.append(distances[nearest_index])
            if len(nearest_ids) == n:
                assignment_ids.append(nearest_ids)
                assignment_distances.append(nearest_distances)
                break

    if n > 1:
        df_com["Segmentation_cells ID"] = assignment_ids
        df_com["Segmentation_cells Distance (nm)"] = assignment_distances
    else:
        df_com["Segmentation_cells ID"] = [ids[0] for ids in assignment_ids]
        df_com["Segmentation_cells Distance (nm)"] = [dists[0] for dists in assignment_distances]
    return df_com


@pytest.mark.parametrize("assignment_type", [0, 1, 2, 3])
def test_assign_to_organelles(
    shared_tmpdir,
    tmp_zarr,
    assignment_type,
):

    atc = AssignToOrganelles(
        organelle_csvs=f"{shared_tmpdir}/csvs/assignment_coms.csv",
        target_organelle_ds_path=f"{tmp_zarr}/segmentation_cells/s0",
        output_path=f"{shared_tmpdir}/csvs/",
        assignment_type=assignment_type,
    )
    atc.get_organelle_assignments()
    test_df = list(atc.organelle_info_dict.values())[0]
    gt_df = gt_assignments(
        f"{shared_tmpdir}/csvs/assignment_coms.csv",
        f"{tmp_zarr}/segmentation_cells/s0",
        assignment_type,
    )
    assert_frame_equal(gt_df, test_df, check_dtype=False)

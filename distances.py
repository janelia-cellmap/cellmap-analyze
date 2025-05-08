# %%
# use kd tree
from scipy import spatial
import pandas as pd
import numpy as np


measurement_file = "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/jrc_22ak351-leaf-3r/plasmodesmata_cleaned_lines.csv"
organelles = pd.read_csv(measurement_file)
com_x = organelles["COM X (nm)"].to_numpy()
com_y = organelles["COM Y (nm)"].to_numpy()
com_z = organelles["COM Z (nm)"].to_numpy()

coms = np.column_stack((com_z, com_y, com_x))

tree = spatial.KDTree(coms)
neighbors = tree.query_ball_tree(tree, 1_000)
densities = np.array([len(n) for n in neighbors])
# densities = densities / densities.max()

# %%
import numpy as np
import cellmap_analyze.util.image_data_interface
from importlib import reload

reload(cellmap_analyze.util.image_data_interface)
import cellmap_analyze.util.image_data_interface
from cellmap_analyze.util.image_data_interface import ImageDataInterface
import cc3d


idi = ImageDataInterface(
    "/groups/cellmap/cellmap/annotations/amira/jrc_22ak351-leaf-3r/crop361/relabeled.zarr/crop361_relabeled/s0",
)
segmented_image = idi.to_ndarray_ts()
connected = cc3d.connected_components(segmented_image, connectivity=6)

# %%
# %timeit boundaries = find_boundaries(segmented_image, mode="outer")

# %%

import numpy as np
from scipy import spatial
from tqdm import tqdm
from cellmap_analyze.cythonizing.process_arrays import find_boundaries

# Set the number of nearest objects you want (n must be > 0).
n = 3  # For example, 3 nearest objects

num_points = len(coms)
boundaries = find_boundaries(connected)
unique_ids = np.unique(boundaries[boundaries > 0])

# Initialize arrays to store the n closest distances and corresponding ids.
# Every query point gets an array of size n.
closest_distances = np.full((num_points, n), np.inf)
closest_ids = np.zeros((num_points, n), dtype=int)

boundary_coords = np.argwhere(boundaries)
boundary_ids = boundaries[
    boundary_coords[:, 0], boundary_coords[:, 1], boundary_coords[:, 2]
]

maximum_distance = 5_000
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
        coords = (boundary_coords[boundary_ids == unique_id] + 0.5) * 128
        tree = spatial.KDTree(coords)

        # Query only for the points (coms) that need updating.
        current_distances, _ = tree.query(
            coms[update_mask], distance_upper_bound=maximum_distance * iteration
        )

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
        sorted_distances = np.take_along_axis(combined_distances, sort_order, axis=1)
        sorted_ids = np.take_along_axis(combined_ids, sort_order, axis=1)

        # Update the candidate arrays for only the query points that needed an update.
        closest_distances[update_mask] = sorted_distances[:, :n]
        closest_ids[update_mask] = sorted_ids[:, :n]


# %%
import matplotlib.pyplot as plt

plt.imshow(boundaries[:, :, 50])
# %%
# %%

# %%
import cc3d

connected = cc3d.connected_components(segmented_image, connectivity=6)
# %%

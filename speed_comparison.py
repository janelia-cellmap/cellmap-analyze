# %%
from importlib import reload
import cellmap_analyze.util.image_data_interface

reload(cellmap_analyze.util.image_data_interface)
from cellmap_analyze.util.image_data_interface import ImageDataInterface

idi = ImageDataInterface(
    "/nrs/cellmap/ackermand/predictions/cellmap_experiments/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.zarr/predictions/2025-02-15_3m/plasmodesmata_affs_lsds/0__affs"
)
from funlib.geometry import Roi

roi = Roi((108 * 8, 108 * 8, 108 * 8), (512 * 8, 512 * 8, 512 * 8)).grow(
    (9 * 8, 9 * 8, 9 * 8), (9 * 8, 9 * 8, 9 * 8)
)

import timeit

ds_data = idi.to_ndarray_ds(roi)
# %timeit ds_data = idi.to_ndarray_ts(roi)

# %%
idi.ds

# %%
ds_data.shape

# %%from importlib import reload
from funlib.geometry import Roi
from cellmap_analyze.util.image_data_interface import ImageDataInterface

idi = ImageDataInterface(
    "/nrs/cellmap/ackermand/predictions/cellmap_experiments/jrc_22ak351-leaf-3r/jrc_22ak351-leaf-3r.zarr/processed/2025-02-15_3r/plasmodesmata_affs_lsds/0__affs_filter_val_0.5_lrb_ratio_-0.08_adj_0.5_lr_1.2_frags/"
)
arr = idi.to_ndarray_ts(Roi((953*8, 3586*8, 6971*8),(512*8,512*8,512*8)))

# %%
import numpy as np
np.unique(arr)
# %%
import numpy as np
from scipy.spatial.distance import cdist

def min_distances_between_nonzero_ids(arr):
    """
    Compute minimum pairwise distances between all distinct nonzero IDs in a 2D NumPy array.
    
    Parameters
    ----------
    arr : numpy.ndarray
        A 2D array with integer IDs. Zeros indicate "no ID."
        
    Returns
    -------
    dict
        A dictionary where keys are (id1, id2) tuples, and values are the minimum distance
        between those two IDs. If either ID does not appear or if no valid distance can be
        computed, the distance will be None.
    """
    # Get all unique nonzero IDs
    unique_ids = np.unique(arr)
    unique_ids = unique_ids[unique_ids != 0]  # exclude 0 from consideration
    
    # Gather the coordinates for each ID
    coords = {}
    for uid in unique_ids:
        coords[uid] = np.argwhere(arr == uid)
    
    # If there's fewer than 2 unique IDs, no pairwise distances exist
    if len(unique_ids) < 2:
        return {}
    
    # Compute minimum distances for each pair of IDs
    results = {}
    for i in range(len(unique_ids)):
        for j in range(i + 1, len(unique_ids)):
            id1 = unique_ids[i]
            id2 = unique_ids[j]
            
            # Coordinates for the two IDs
            c1 = coords[id1]
            c2 = coords[id2]
            
            # If either set of coordinates is empty (unlikely, but just in case), skip
            if c1.size == 0 or c2.size == 0:
                results[(id1, id2)] = None
                continue
            
            # Compute all pairwise distances
            dist_matrix = cdist(c1, c2, metric='euclidean')
            
            # Minimum distance among all pairs of points from id1 and id2
            min_dist = np.min(dist_matrix)
            
            results[(id1, id2)] = min_dist
    
    return results



distances = min_distances_between_nonzero_ids(arr)
for pair_ids, dist in distances.items():
    print(f"IDs {pair_ids}: min distance = {dist:.2f}")
# %%

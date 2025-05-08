# %%
from cellmap_analyze.process.mutex_watershed import MutexWatershed

mw = MutexWatershed(
    "/tmp/pytest-of-ackermand/pytest-current/tmpcurrent/tmp.zarr/affinities_cylinders",
    "./test.zarr",
    adjacent_edge_bias=0,
    lr_bias_ratio=0,
    filter_val=0,
    connectivity=1,
    num_workers=1
)

mw.get_connected_components()
# %%
# holes
from funlib.geometry import Roi
from cellmap_analyze.util.image_data_interface import ImageDataInterface
import numpy as np
from skimage.segmentation import find_boundaries
from cellmap_analyze.process.connected_components import ConnectedComponents

connectivity = 1
holes_idi = ImageDataInterface(
    "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-comma-1/jrc_c-elegans-comma-1.zarr/lyso_filled_holes/s0"
)
input_idi = ImageDataInterface(
    "/nrs/cellmap/ackermand/cellmap/c-elegans/jrc_c-elegans-comma-1/jrc_c-elegans-comma-1.zarr/lyso/s0"
)
roi = Roi((np.array((39342, 8715, 19275)) // 16) * 16, [212 * 16] * 3)
holes = holes_idi.to_ndarray_ts(roi)
input = input_idi.to_ndarray_ts(roi)

max_input_id = np.max(input)

input_boundaries = find_boundaries(input, mode="inner").astype(np.uint64)
hole_boundaries = find_boundaries(holes, mode="inner").astype(np.uint64)

holes = holes.astype(np.uint64)
holes[holes > 0] += max_input_id  # so there is no id overlap
data = holes + input
mask = np.logical_or(input_boundaries, hole_boundaries)
touching_ids = ConnectedComponents.get_touching_ids(data, mask, connectivity)

hole_to_object_dict = {}
for id1, id2 in touching_ids:
    if id2 <= max_input_id:
        continue
    id2 -= max_input_id
    # then input objects are touching holes
    object_ids = hole_to_object_dict.get(id2, set())
    object_ids.add(id1)
    hole_to_object_dict[id2] = object_ids
print(hole_to_object_dict)
# %%
# skeletonization
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.morphology import skeletonize, skeletonize_3d
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert

image = invert(data.horse()) * 55

skeleton = skeletonize(image)
skeleton_3d = skeletonize_3d(image)

print(skeleton.max())
# take maximum along axis
plt.imshow(skeleton_3d.max(axis=0))

# %%

# %%
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label
from skimage import io  # or use nibabel, SimpleITK, etc. to load your 3D data
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from funlib.geometry import Roi
import edt

binary = (
    ImageDataInterface(
        "/nrs/cellmap/ackermand/cellmap/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/canaliculi_filteredIDs_cc_close_raw_mask_filled/s0"
    ).to_ndarray_ts(Roi((0, 0, 0), (500 * 128, 500 * 128, 500 * 128)))
    > 0
)
print("Read data")

# 2. Compute the Euclidean distance transform
distance = ndi.distance_transform_edt(binary)
print("distanced")
# 3. Find local maxima in the distance map to serve as watershed markers
#    - footprint defines the neighborhood for maxima detection (3×3×3 here)
#    - threshold_rel filters out very small peaks (<10% of max distance)
coords = peak_local_max(
    distance, footprint=np.ones((20, 20, 20)), threshold_rel=0.5, labels=binary
)
print("coords")

# 4. Build a marker image (same shape as distance) where each peak gets a unique integer label
markers = np.zeros_like(distance, dtype=np.int32)
for idx, (z, y, x) in enumerate(coords, start=1):
    markers[z, y, x] = idx
print("markers")

# 5. Run the watershed on the inverted distance map, constrained by your binary mask
labels = watershed(
    image=-distance,  # we flood “downslope” from peaks, so invert
    markers=markers,  # seed labels
    mask=binary,  # only segment within the original foreground
)
print("watershed")

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops

# Generate an initial image with two overlapping circles
x, y = np.indices((100, 100))
x1, y1, x2, y2 = 15, 15, 44, 52
r1, r2 = 14, 40
mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1**2
mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2**2
image = np.zeros((100, 100), dtype=np.uint8)
image[mask_circle1] = 1
image[mask_circle2] = 2


# image = np.logical_or(mask_circle1, mask_circle2)

for i in range(2):
    if i == 1:
        # second iteration: crop the image
        image = image[:-76, :-76]

    # 1. Distance transform
    distance = ndi.distance_transform_edt(image)

    # 2. Boolean mask of all local‐max pixels
    coords = peak_local_max(
        distance, footprint=np.ones((2, 2), dtype=bool), labels=image
    )
    plateau_mask = np.zeros_like(distance, dtype=bool)
    plateau_mask[tuple(coords.T)] = True

    # 3. Label each connected plateau region
    plateau_labels = label(plateau_mask, connectivity=1)

    # # 4. Compute centroids and build a new seed mask
    # seed_mask = np.zeros_like(plateau_mask)
    # for prop in regionprops(plateau_labels):
    #     y, x = prop.centroid
    #     seed_mask[int(round(y)), int(round(x))] = True

    # # 5. Label your one‐pixel seeds
    # markers, _ = ndi.label(seed_mask)

    # 6. Run watershed
    # labels = watershed(-distance, markers, mask=image)
    labels = watershed(-distance, markers=plateau_labels, mask=image)
    # Plot
    fig, axes = plt.subplots(ncols=4, figsize=(9, 4), sharex=True, sharey=True)
    ax0, ax1, ax2, ax3 = axes

    # 1) Overlapping + seeds
    ax0.imshow(image, cmap=plt.cm.gray)
    ax0.scatter(
        coords[:, 1],
        coords[:, 0],
        s=30,
        edgecolor="red",
        facecolor="none",
        linewidth=1.2,
        label="seeds",
    )
    ax0.set_title("Overlapping objects\n(with seeds)")
    ax0.legend(loc="upper right", markerscale=0.7)

    # 2) Inverted distance map
    ax1.imshow(-distance, cmap=plt.cm.gray)
    ax1.set_title("−Distance")

    ax2.imshow(plateau_labels, cmap=plt.cm.nipy_spectral, interpolation="none")
    ax2.set_title("labels")

    # 3) Watershed result
    ax3.imshow(labels, cmap=plt.cm.nipy_spectral, interpolation="none")
    ax3.set_title("Separated objects")

    for ax in (ax0, ax1, ax2):
        ax.set_axis_off()

    fig.tight_layout()
    plt.show()
    break

# %%
image = np.zeros((100, 100, 100), dtype=np.uint8)
z, y, x = np.indices((100, 100, 100))
z1, y1, x1, z2, y2, x2, z3, y3, x3 = 15, 15, 15, 44, 52, 55, 70, 70, 70
r1, r2, r3 = 14, 40, 15
mask_circle1 = (z - z1) ** 2 + (x - x1) ** 2 + (y - y1) ** 2 < r1**2
mask_circle2 = (z - z2) ** 2 + (x - x2) ** 2 + (y - y2) ** 2 < r2**2
image = np.zeros((100, 100, 100), dtype=np.uint8)
image[mask_circle1] = 1
image[mask_circle2] = 2

# %%

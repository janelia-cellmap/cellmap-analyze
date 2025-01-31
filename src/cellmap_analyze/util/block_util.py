import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

# SELEM = np.ones((3, 3, 3), dtype=bool)


def erosion(block, iterations, structuring_element):
    # Erode each region, has to be done like this in case regions touch
    eroded_image = np.zeros_like(block)
    for id in np.unique(block):
        if id == 0:  # Skip background
            continue
        mask = block == id
        eroded_mask = binary_erosion(
            mask, structure=structuring_element, iterations=iterations
        )
        eroded_image[eroded_mask] = id
    block = eroded_image
    return block


def dilation(block, iterations, structuring_element):
    block = binary_dilation(
        block > 0, structure=structuring_element, iterations=iterations
    )
    return block


# %%
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

SELEM = np.ones((3, 3, 3), dtype=bool)
a = np.zeros((7, 7), dtype=int)
a[:6, :5] = 1
binary_erosion(a, iterations=1)

# %%

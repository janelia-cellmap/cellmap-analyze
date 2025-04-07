import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.dask_util import DaskBlock
import fastremap

# SELEM = np.ones((3, 3, 3), dtype=bool)

# Probably want to replace this by fastmorph


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


def relabel_block(
    block: DaskBlock,
    input_idi: ImageDataInterface,
    output_idi: ImageDataInterface,
    mask=None,
):
    # All ids must be accounted for in the relabeling dict
    data = input_idi.to_ndarray_ts(
        block.write_roi,
    )
    if mask:
        mask_block = mask.process_block(roi=block.write_roi)
        data *= mask_block

    if len(block.relabeling_dict) > 0:
        try:
            fastremap.remap(
                data, block.relabeling_dict, preserve_missing_labels=True, in_place=True
            )
        except:
            raise Exception(
                f"Error in relabel_block {block.write_roi}, {list(block.relabeling_dict.keys())}, {list(block.relabeling_dict.values())}"
            )

    output_idi.ds[block.write_roi] = data

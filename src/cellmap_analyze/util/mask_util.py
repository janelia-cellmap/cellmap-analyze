# %%
from scipy import ndimage
import cellmap_analyze.util.image_data_interface
from importlib import reload
import cellmap_analyze.util.block_util

reload(cellmap_analyze.util.image_data_interface)
reload(cellmap_analyze.util.block_util)
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.block_util import erosion, dilation
from functools import partial
from funlib.geometry import Roi, Coordinate


class Mask:
    def __init__(
        self,
        path,
        mask_type="exclusive",
        operation="simple",
        iterations=0,
        output_voxel_size=None,
        connectivity=2,
        mask_value=None,
    ):
        if type(operation) == str:
            operation = [operation]
        if type(iterations) == int:
            iterations = [iterations]

        if len(iterations) != len(operation):
            raise ValueError("Iterations and operation must have the same length")

        structuring_element = ndimage.generate_binary_structure(3, connectivity)
        if "erosion" in operation:
            self.idi = ImageDataInterface(
                path,
                output_voxel_size=output_voxel_size,
                custom_fill_value="edge",
            )
        else:
            self.idi = ImageDataInterface(path, output_voxel_size=output_voxel_size)
        self.output_voxel_size = output_voxel_size
        if not self.output_voxel_size:
            self.output_voxel_size = self.idi.voxel_size

        if (not operation and iterations > 0) or (operation and iterations == 0):
            raise ValueError(
                "Iterations must be set if operation is set and vice versa"
            )

        self.process_block = partial(
            self._process_block,
            operation=operation,
            mask_type=mask_type,
            iterations=iterations,
            mask_value=mask_value,
            structuring_element=structuring_element,
        )

    def _process_block(
        self, operation, mask_type, iterations, mask_value, structuring_element, roi
    ):
        if not roi:
            roi = self.idi.roi

        if operation == ["simple"]:
            block = self.idi.to_ndarray_ts(roi)
        else:
            total_iterations = sum(iterations)
            padding = total_iterations * self.output_voxel_size[0]
            block = self.idi.to_ndarray_ts(
                roi.grow(
                    padding,
                    padding,
                )
            )
            if mask_value is not None:
                block = block == mask_value
            for operation, iterations in zip(operation, iterations):
                if operation == "erosion":
                    block = erosion(block, iterations, structuring_element)
                else:
                    block = dilation(block, iterations, structuring_element)

            block = block[
                total_iterations:-total_iterations,
                total_iterations:-total_iterations,
                total_iterations:-total_iterations,
            ]

        if mask_type == "exclusive":
            block = block == 0
        else:
            block = block > 0

        return block


class MasksFromConfig:
    def __init__(self, mask_config_dict, output_voxel_size, connectivity=2):
        self.connectivity = connectivity
        self.mask_dict = {}
        for mask_name, mask_config in mask_config_dict.items():
            self.mask_dict[mask_name] = Mask(
                **mask_config,
                output_voxel_size=output_voxel_size,
                connectivity=connectivity,
            )

    def process_block(self, roi):
        for idx, mask in enumerate(self.mask_dict.values()):
            if idx == 0:
                block = mask.process_block(roi=roi)
            else:
                block &= mask.process_block(roi=roi)

        return block


# mask_config = {
#     "mask_one": {
#         "path": "/tmp/pytest-of-ackermand/pytest-current/tmp0/tmp.zarr/mask_one/s0",
#         "operation": "erosion",
#         "mask_type": "exclusive",
#         "iterations": 2,
#     },
#     "mask_two": {
#         "path": "/tmp/pytest-of-ackermand/pytest-current/tmp0/tmp.zarr/mask_two/s0",
#         "mask_type": "exclusive",
#         "operation": "dilation",
#         "iterations": 1,
#     },
# }
# import numpy as np

# roi = Roi((0, 0, 0), (88, 88, 88))
# masks = MasksFromConfig(mask_config, output_voxel_size=Coordinate(8, 8, 8))
# block = masks.process_block(roi).astype(np.uint8)
# from cellmap_analyze.util.visualization_util import view_in_neuroglancer

# raw = ImageDataInterface(
#     "/nrs/cellmap/data/jrc_mus-liver-zon-2/jrc_mus-liver-zon-2.zarr/recon-1/em/fibsem-uint8/s4"
# )
# view_in_neuroglancer(
#     block=block,
#     mask_one=masks.mask_dict["mask_one"].idi.to_ndarray_ts(roi),
#     mask_two=masks.mask_dict["mask_two"].idi.to_ndarray_ts(roi),
# )


mask_config = {
    "cells": {
        "path": "/nrs/cellmap/data/jrc_mus-liver-zon-2/jrc_mus-liver-zon-2.zarr/recon-1/labels/inference/segmentations/cell/s0",
        "operation": "erosion",
        "mask_type": "exclusive",
        "iterations": 4,
    },
    "ecs": {
        "path": "/nrs/cellmap/zouinkhim/liver_zonation/jrc_mus-liver-zon-2_postprocessed_uint8.zarr/s5",
        "mask_type": "exclusive",
        "operation": ["erosion", "dilation"],
        "iterations": [3, 5],
    },
    "raw": {
        "path": "/nrs/cellmap/data/jrc_mus-liver-zon-2/jrc_mus-liver-zon-2.zarr/recon-1/em/fibsem-uint8/s4",
        "mask_type": "exclusive",
        "mask_value": 0,
        "operation": "dilation",
        "iterations": 10,
    },
}
import numpy as np

roi = Roi((32160, 40385, 19425), (128 * 200, 128 * 200, 128 * 200))
masks = MasksFromConfig(mask_config, output_voxel_size=Coordinate(128, 128, 128))
block = masks.process_block(roi).astype(np.uint8)
from cellmap_analyze.util.visualization_util import view_in_neuroglancer

raw = ImageDataInterface(
    "/nrs/cellmap/data/jrc_mus-liver-zon-2/jrc_mus-liver-zon-2.zarr/recon-1/em/fibsem-uint8/s4"
)
view_in_neuroglancer(
    raw=raw.to_ndarray_ts(roi),
    block=block,
    cells=masks.mask_dict["cells"].idi.to_ndarray_ts(roi),
    ecs=masks.mask_dict["ecs"].idi.to_ndarray_ts(roi),
    thresholded=masks.mask_dict["raw"].process_block(roi=roi).astype(np.uint8),
)

# %%
# import numpy as np

# mask = Mask(
#     "/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/labels/inference/segmentations/cell/s0",
#     mask_type="exclusive",
#     operation="erosion",
#     iterations=1,
#     output_voxel_size=Coordinate(8, 8, 8),
# )
# block = mask.idi.to_ndarray_ts(roi)
# block_erosion = mask.process_block(roi=roi).astype(np.uint8)
# print(np.unique(block))
# difference = block.astype(np.uint8) - block_erosion.astype(np.uint8)
# from cellmap_analyze.util.visualization_util import view_in_neuroglancer

# view_in_neuroglancer(block=block, block_erosion=block_erosion, difference=difference)

# # %%
# mask.idi.ds.shape
# %%

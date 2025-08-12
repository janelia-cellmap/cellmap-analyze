# %%
from scipy import ndimage
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.block_util import erosion, dilation
from functools import partial


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
        chunk_shape=None,
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
                chunk_shape=chunk_shape,
            )
        else:
            self.idi = ImageDataInterface(
                path, output_voxel_size=output_voxel_size, chunk_shape=chunk_shape
            )
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
            if mask_value is not None:
                block = block == mask_value
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

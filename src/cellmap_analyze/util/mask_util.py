# %%
import numpy as np
from scipy import ndimage
from cellmap_analyze.util.grid_util import grid_of
from cellmap_analyze.util.zarr_util import open_dataset
from cellmap_analyze.util.block_util import erosion, dilation
from functools import partial


def _apply_mask_value(block, mask_value):
    """``mask_value`` is either a scalar (exact match, ``block == mask_value``)
    or a ``[min, max]`` pair (inclusive range, ``min <= block <= max``) --
    the range form lets a raw-intensity mask express a ceiling/floor (e.g.
    excluding saturated/over-threshold voxels) without a separate precomputed
    mask volume.
    """
    if isinstance(mask_value, (list, tuple)):
        lo, hi = mask_value
        return (block >= lo) & (block <= hi)
    return block == mask_value


class Mask:
    def __init__(
        self,
        path,
        mask_type="exclusive",
        operation="simple",
        iterations=0,
        output_grid=None,
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
        self.vol = open_dataset(path, padding="edge" if "erosion" in operation else 0)
        # read on the caller's grid
        self.grid = grid_of(self.vol) if output_grid is None else output_grid

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
        self,
        begin=None,
        end=None,
        *,
        operation,
        mask_type,
        iterations,
        mask_value,
        structuring_element,
    ):
        """The mask over voxels ``[begin, end)`` of the caller's grid (all of it when
        None)."""
        if begin is None:
            begin, end = (0,) * 3, self.grid.shape

        if operation == ["simple"]:
            block = self.vol.read_onto(self.grid, begin, end)
            if mask_value is not None:
                block = _apply_mask_value(block, mask_value)
        else:
            # each iteration reaches one voxel further on every axis
            padding = sum(iterations)
            block = self.vol.read_onto(
                self.grid, np.subtract(begin, padding), np.add(end, padding)
            )

            if mask_value is not None:
                block = _apply_mask_value(block, mask_value)
            for operation, iterations in zip(operation, iterations):
                if operation == "erosion":
                    block = erosion(block, iterations, structuring_element)
                else:
                    block = dilation(block, iterations, structuring_element)

            if padding:
                block = block[(slice(padding, -padding),) * 3]

        if mask_type == "exclusive":
            block = block == 0
        else:
            block = block > 0

        return block


class MasksFromConfig:
    def __init__(self, mask_config_dict, output_grid, connectivity=2):
        self.connectivity = connectivity
        self.mask_dict = {}
        for mask_name, mask_config in mask_config_dict.items():
            self.mask_dict[mask_name] = Mask(
                **mask_config,
                output_grid=output_grid,
                connectivity=connectivity,
            )

    def process_block(self, begin=None, end=None):
        for idx, mask in enumerate(self.mask_dict.values()):
            if idx == 0:
                block = mask.process_block(begin, end)
            else:
                block &= mask.process_block(begin, end)

        return block

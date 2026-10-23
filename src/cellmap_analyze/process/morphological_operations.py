# %%
import numpy as np
from scipy import ndimage
from cellmap_analyze.util import dask_util
from cellmap_analyze.util.block_util import erosion
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
)
from cellmap_analyze.util.grid_util import crop_to_roi, grid_of
from cellmap_analyze.util.mask_util import MasksFromConfig

import logging
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.zarr_util import create_multiscale_dataset, open_dataset

import fastmorph

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


_VALID_OPERATIONS = ("erosion", "dilation", "opening", "closing")


class MorphologicalOperations(ComputeConfigMixin):
    """Blockwise morphological cleanup of a labeled (multi-instance)
    segmentation via ``fastmorph``, which -- in its default multilabel mode,
    used throughout this class -- treats every voxel's existing label as a
    hard boundary: dilation only ever claims *background* voxels (majority
    vote among neighboring labels), so no operation here can merge or bleed
    across two distinct instance ids. That's what makes it safe to run
    directly on a full labeled dataset rather than needing to isolate each
    instance first.

    ``opening`` (erode then dilate, ``iterations`` each) strips small
    protrusions/isolated specks; ``closing`` (dilate then erode) fills small
    pits and non-enclosed gaps that a topological hole-filler can't reach.
    Blockwise correctness for all four operations rests on a halo argument:
    ``fastmorph``'s ``iterations=N`` op has an exact Chebyshev-radius-N
    dependency (N sequential 1-voxel-stencil passes), so reading each block
    with an N-voxel halo (2N for the two-pass composite ops, since each pass
    contributes its own radius) and trimming that same (possibly asymmetric,
    near the true dataset boundary -- see the halo-construction comment in
    ``perform_morphological_operation_blockwise``) margin off the result
    reproduces exactly what a monolithic whole-dataset run would produce,
    including right at the dataset's true edge. Verified empirically: with a
    naive zero-filled-beyond-the-edge halo instead, a 3-instance dataset
    with an isolated single-voxel fleck at the true corner mismatched the
    whole-array reference at a handful of voxels there (fastmorph's
    multilabel dilation resolves contested background voxels by a majority
    vote among neighbors, and a phantom extra background neighbor can tip a
    close vote) -- not reading past the true edge at all fixes this exactly.
    """

    def __init__(
        self,
        input_path,
        output_path,
        mask_config=None,
        num_workers=10,
        roi=None,
        chunk_shape=None,
        operation="erosion",
        iterations=1,
        connectivity=2,
    ):
        super().__init__(num_workers)

        self.input_vol = open_dataset(input_path)
        self.grid = crop_to_roi(grid_of(self.input_vol, chunk_shape), roi)

        self.mask = None
        if mask_config:
            self.mask = MasksFromConfig(
                mask_config,
                output_grid=self.grid,
                connectivity=connectivity,
            )

        if operation not in _VALID_OPERATIONS:
            raise ValueError(
                f"operation must be one of {_VALID_OPERATIONS}, got {operation!r}"
            )
        self.operation = operation
        if iterations < 1:
            raise ValueError("iterations must be at least 1")

        self.iterations = iterations
        self.output_vol = create_multiscale_dataset(
            output_path, dtype=self.input_vol.dtype, grid=self.grid
        )

    @staticmethod
    def perform_morphological_operation_blockwise(
        block_index,
        grid,
        input_vol,
        output_vol,
        operation: str,
        iterations: int,
        mask: MasksFromConfig = None,
    ):
        # Composite (two-pass) ops need a halo covering both passes' context
        # -- each fastmorph iteration has an exact 1-voxel dependency radius,
        # so N iterations need an N-voxel halo, and two chained N-iteration
        # passes need 2N (see class docstring for the full argument).
        padding_voxels = iterations * (2 if operation in ("opening", "closing") else 1)
        # read_beyond_roi=False: don't manufacture phantom zero-valued
        # neighbors past the true dataset edge. fastmorph's multilabel
        # dilation resolves contested background voxels by a majority vote
        # among neighbors, and an explicit (but fake) extra background
        # neighbor there can tip a close vote away from what a genuine
        # whole-dataset run would produce -- confirmed empirically (a
        # 3-instance dataset with an isolated fleck at the true corner
        # mismatched the whole-array reference at 3 voxels with
        # read_beyond_roi=True, iterations=2, and matched exactly once this
        # was set to False). Near the true edge the halo is then narrower
        # than padding_voxels on that side -- exactly like a genuine
        # whole-array call, which has no context beyond its own edge either
        # -- so only the block's own voxels (block.inner) are written.
        block = create_block_from_index(
            grid,
            block_index,
            padding=padding_voxels,
            read_beyond_roi=False,
        )
        if mask:
            mask_block = mask.process_block(block.read_begin, block.read_end)
            if not np.any(mask_block):
                output_vol.write_index(block.begin, block.end, 0)
                return

        data = input_vol.read_onto(grid, block.read_begin, block.read_end)
        if mask:
            data *= mask_block

        if np.any(data):
            if operation == "erosion":
                data = fastmorph.erode(data, iterations=iterations)
            elif operation == "dilation":
                data = fastmorph.dilate(data, iterations=iterations)
            elif operation == "opening":
                data = fastmorph.erode(data, iterations=iterations)
                data = fastmorph.dilate(data, iterations=iterations)
            elif operation == "closing":
                data = fastmorph.dilate(data, iterations=iterations)
                data = fastmorph.erode(data, iterations=iterations)

            if mask:
                # need before and after to make sure nothing from outside mask makes it in and vice versa
                data *= mask_block
        # else: an all-background block stays all-background under every one
        # of these ops -- skip straight to the (still all-zero) write below.

        output_vol.write_index(block.begin, block.end, data[block.inner])

    def perform_morphological_operation(self):
        num_blocks = dask_util.get_num_blocks(self.grid)
        dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            f"{self.operation} of {self.input_vol.level_path}",
            MorphologicalOperations.perform_morphological_operation_blockwise,
            self.grid,
            self.input_vol,
            self.output_vol,
            self.operation,
            self.iterations,
            self.mask,
        )

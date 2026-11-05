import itertools
import logging
import os
import shutil

import numpy as np

from cellmap_analyze.cythonizing.topological_thinning import (
    faces_are_final,
    halo_for,
    thin,
)
from cellmap_analyze.util import dask_util
from cellmap_analyze.util.dask_util import create_block_from_index
from cellmap_analyze.util.grid_util import crop_to_roi, grid_of
from cellmap_analyze.util.io_util import TimingMessager, split_dataset_path
from cellmap_analyze.util.mask_util import MasksFromConfig
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.zarr_util import create_multiscale_dataset, open_dataset

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# what a block does in a pass
RUN, COPY = 0, 1
# a block's state between passes
ACTIVE, DONE, FINISHED = 0, 1, 2


def _unsigned(data):
    """``data`` viewed as the unsigned integers the kernel thins."""
    if data.dtype == bool:
        return data.view(np.uint8)
    if data.dtype.kind == "i":
        return data.view(data.dtype.str.replace("i", "u"))
    if data.dtype.kind != "u":
        raise TypeError(f"cannot thin a {data.dtype} segmentation")
    return data


def deletion_bands(deleted, band):
    """Where in a block its deletions lie, as 27 bits: bit ``9 i + 3 j + k``
    is set when one lies, along z, y and x, within ``band`` of the block's low
    face (0), anywhere (1) or within ``band`` of its high face (2)."""
    coords = np.nonzero(deleted)
    if len(coords[0]) == 0:
        return 0
    codes = 0
    for axis, c in enumerate(coords):
        n = deleted.shape[axis]
        code = (c < band).astype(np.int64) | ((c >= n - band).astype(np.int64) << 1)
        codes = codes * 4 + code
    bands = 0
    for combined in np.unique(codes):
        per_axis = [(int(combined) >> shift) & 3 for shift in (4, 2, 0)]
        choices = [
            [1] + ([0] if code & 1 else []) + ([2] if code & 2 else [])
            for code in per_axis
        ]
        for i, j, k in itertools.product(*choices):
            bands |= 1 << (9 * i + 3 * j + k)
    return bands


def blocks_near_deletions(bands):
    """For a grid of blocks' ``deletion_bands``, which blocks have a deletion
    in a 26-neighbour within the band of their shared face, edge or corner."""
    near = np.zeros(bands.shape, dtype=bool)
    # a neighbour below along an axis matters through its high band (2),
    # one above through its low band (0), one level with it anywhere (1)
    band_of = {-1: 2, 0: 1, 1: 0}
    for delta in itertools.product((-1, 0, 1), repeat=3):
        if delta == (0, 0, 0):
            continue
        bit = 1 << (9 * band_of[delta[0]] + 3 * band_of[delta[1]] + band_of[delta[2]])
        has = (bands & bit) != 0
        receiver, sender = [], []
        for d in delta:
            receiver.append(slice(None, -1) if d == 1 else slice(1, None) if d == -1 else slice(None))
            sender.append(slice(1, None) if d == 1 else slice(None, -1) if d == -1 else slice(None))
        near[tuple(receiver)] |= has[tuple(sender)]
    return near


class TopologicalThinning(ComputeConfigMixin):
    """Blockwise Lee (1994) thinning of a segmentation to curve skeletons or,
    with ``medial_surface``, medial surfaces: a port of COSEM's
    ``SparkTopologicalThinning`` whose result is the same, voxel for voxel, as
    thinning the whole (zero-padded) volume at once. Each object is thinned
    as if it were alone, so objects that touch do not fuse, and the output
    keeps their IDs.

    One iteration of the kernel (6 border directions, each re-checked over 8
    parity subfields of the dataset's own voxel index) only reaches 48 voxels,
    so a pass of ``iterations_per_pass`` iterations reads each block with a
    ``48 * iterations_per_pass + 2`` voxel halo from the previous pass's
    output and writes its interior to the other of two buffers. Blocks with
    no deletions of their own or within that halo in the last pass are
    skipped. A block whose face voxels can no longer be deleted (end points,
    or surface end points for medial surfaces) no longer depends on its
    neighbours, so it is thinned to completion with no halo and retired. The
    parity is taken from the dataset's own grid, so a ``roi`` thins as a crop
    of the dataset would (voxels outside it are background).

    More iterations per pass mean fewer reads and writes per iteration for
    bigger blocks (the halo must fit in a block), but more work per pass. ER
    at 8 nm, a few voxels thick, takes a few iterations; thick objects take
    about their inscribed radius in voxels, so skeletonize those at a coarser
    resolution.
    """

    def __init__(
        self,
        segmentation_path,
        output_path,
        medial_surface=False,
        iterations_per_pass=1,
        block_shape_voxels=None,
        block_multiplier=4,
        mask_config=None,
        roi=None,
        num_workers=10,
        chunk_shape=None,
        delete_tmp=True,
    ):
        super().__init__(num_workers)
        self.segmentation_path = segmentation_path
        self.segmentation_vol = open_dataset(segmentation_path)
        self.output_path = str(output_path).rstrip("/")
        os.makedirs(split_dataset_path(self.output_path)[0], exist_ok=True)

        self.medial_surface = bool(medial_surface)
        self.iterations_per_pass = int(iterations_per_pass)
        if self.iterations_per_pass < 1:
            raise ValueError("iterations_per_pass must be at least 1")
        self.halo = halo_for(self.iterations_per_pass)

        dataset_grid = grid_of(self.segmentation_vol, chunk_shape)
        self.grid = crop_to_roi(dataset_grid, roi)
        chunks = np.array(self.grid.chunks)
        if block_shape_voxels is not None:
            self.block_shape = tuple(int(b) for b in block_shape_voxels)
        else:
            self.block_shape = tuple(int(c) * int(block_multiplier) for c in chunks)
        if np.any(np.array(self.block_shape) % chunks):
            raise ValueError(
                f"block shape {self.block_shape} must be a multiple of the chunks {tuple(chunks)}"
            )
        if self.halo > min(self.block_shape):
            raise ValueError(
                f"{self.iterations_per_pass} iterations per pass need a {self.halo} voxel halo, "
                f"more than the block shape {self.block_shape}; use bigger blocks or fewer iterations"
            )
        # the global index of the grid's first voxel fixes the subfield order
        self.parity_origin = np.rint(
            (np.array(self.grid.corner) - np.array(dataset_grid.corner))
            / np.array(self.grid.voxel_size)
        ).astype(int)
        self.num_blocks = dask_util.get_num_blocks(self.grid, self.block_shape)
        self.block_grid_shape = tuple(
            int(np.ceil(s / b)) for s, b in zip(self.grid.shape, self.block_shape)
        )

        self.mask = None
        if mask_config:
            self.mask = MasksFromConfig(mask_config, output_grid=self.grid, connectivity=2)

        self.delete_tmp = delete_tmp
        dtype = self.segmentation_vol.dtype
        self.buffer_paths = [f"{self.output_path}_tmp_pass_{p}" for p in ("even", "odd")]
        self.buffers = [
            create_multiscale_dataset(path, dtype=dtype, grid=self.grid)
            for path in self.buffer_paths
        ]

    @staticmethod
    def process_block(
        task,
        grid,
        block_shape,
        halo,
        source_vol,
        output_vol,
        first_pass,
        mask,
        medial_surface,
        iterations,
        parity_origin,
    ):
        block_index, action = task
        if action == COPY:
            block = create_block_from_index(grid, block_index, block_shape=block_shape)
            data = source_vol.read_onto(grid, block.begin, block.end)
            output_vol.write_index(block.begin, block.end, data)
            return {"index": block_index, "deleted": 0, "bands": 0, "done": False, "empty": False}

        block = create_block_from_index(grid, block_index, padding=halo, block_shape=block_shape)
        read_begin, read_end = np.array(block.read_begin), np.array(block.read_end)
        data = source_vol.read_onto(grid, read_begin, read_end)
        dtype = data.dtype
        data = np.ascontiguousarray(_unsigned(data))
        if first_pass:
            if mask:
                data = data * mask.process_block(read_begin, read_end).astype(data.dtype)
            # voxels outside the grid (a roi) are background
            inside = tuple(
                slice(max(0, -b), min(e - b, s - b))
                for b, e, s in zip(read_begin, read_end, grid.shape)
            )
            cropped = np.zeros_like(data)
            cropped[inside] = data[inside]
            data = cropped

        before = data[block.inner].copy()
        if first_pass and not before.any():
            return {"index": block_index, "deleted": 0, "bands": 0, "done": True, "empty": True}

        thin(data, tuple(int(o) for o in parity_origin + read_begin), medial_surface, iterations)
        lo = np.array(block.begin) - read_begin
        hi = lo + np.array(block.end) - np.array(block.begin)
        done = faces_are_final(data, tuple(int(v) for v in lo), tuple(int(v) for v in hi), medial_surface)
        interior = np.ascontiguousarray(data[block.inner])
        if done:
            # its face voxels are final, so the rest thins without its neighbours
            thin(interior, tuple(int(o) for o in parity_origin + np.array(block.begin)), medial_surface, 0)

        deleted = (before != 0) & (interior == 0)
        output_vol.write_index(block.begin, block.end, interior.view(dtype))
        return {
            "index": block_index,
            "deleted": int(deleted.sum()),
            "bands": deletion_bands(deleted, halo),
            "done": bool(done),
            "empty": False,
        }

    @staticmethod
    def _process_partition(tasks, *args):
        return [TopologicalThinning.process_block(task, *args) for task in tasks]

    def _run_pass(self, tasks, pass_index):
        source = self.segmentation_vol if pass_index == 0 else self.buffers[(pass_index - 1) % 2]
        partitions = dask_util.block_partitions(len(tasks), self.num_workers)
        return dask_util.compute_partitions(
            [[tasks[i] for i in part] for part in partitions],
            TopologicalThinning._process_partition,
            self.grid,
            self.block_shape,
            self.halo,
            source,
            self.buffers[pass_index % 2],
            pass_index == 0,
            self.mask,
            self.medial_surface,
            self.iterations_per_pass,
            self.parity_origin,
            compute_args=self.compute_args,
        )

    def thin(self):
        """Thin every block until no iteration deletes anything, then move the
        last pass's buffer to ``output_path``."""
        kind = "medial surface" if self.medial_surface else "skeleton"
        state = np.full(self.num_blocks, ACTIVE, dtype=np.uint8)
        tasks = [(i, RUN) for i in range(self.num_blocks)]
        pass_index = 0
        with dask_util.start_dask(self.num_workers, f"topological thinning ({kind})", logger):
            while True:
                num_runs = sum(1 for _, action in tasks if action == RUN)
                with TimingMessager(
                    f"Thinning pass {pass_index} ({kind}, {self.iterations_per_pass} "
                    f"iteration(s)): {num_runs} blocks to thin, {len(tasks) - num_runs} to copy",
                    logger,
                ):
                    results = self._run_pass(tasks, pass_index)

                deleted = np.zeros(self.num_blocks, dtype=np.int64)
                bands = np.zeros(self.num_blocks, dtype=np.int64)
                for result in results:
                    i = result["index"]
                    deleted[i] = result["deleted"]
                    bands[i] = result["bands"]
                    if result["empty"]:
                        state[i] = FINISHED
                    elif result["done"] and state[i] == ACTIVE:
                        state[i] = DONE
                    elif state[i] == DONE:
                        # copied into the other buffer, so both hold it
                        state[i] = FINISHED

                rerun = (deleted > 0) | blocks_near_deletions(
                    bands.reshape(self.block_grid_shape)
                ).ravel()
                run = (state == ACTIVE) & rerun
                logger.info(
                    "Pass %d deleted %d voxels; %d blocks to rethin, %d retired",
                    pass_index,
                    int(deleted.sum()),
                    int(run.sum()),
                    int((state != ACTIVE).sum()),
                )
                if not run.any():
                    break
                # The other buffer must hold every block that is skipped: blocks
                # retired this pass, and after the first pass all blocks.
                copy = state == DONE
                if pass_index == 0:
                    copy |= (state == ACTIVE) & ~rerun
                tasks = [(int(i), RUN) for i in np.flatnonzero(run)]
                tasks += [(int(i), COPY) for i in np.flatnonzero(copy)]
                pass_index += 1

        self._finalize(pass_index)

    def _finalize(self, last_pass):
        final_path = self.buffer_paths[last_pass % 2]
        other_path = self.buffer_paths[1 - last_pass % 2]
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.rename(final_path, self.output_path)
        if self.delete_tmp:
            dask_util.delete_tmp_dir_blockwise(
                f"{other_path}/s0", self.num_workers, self.compute_args
            )
        self.output_vol = open_dataset(f"{self.output_path}/s0")
        logger.info("Wrote %s after %d passes", self.output_path, last_pass + 1)

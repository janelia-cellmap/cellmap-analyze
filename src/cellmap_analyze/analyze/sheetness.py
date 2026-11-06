import logging
import os

import edt as edt_module
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter

from cellmap_analyze.cythonizing.geodesic import nearest_source_values
from cellmap_analyze.util import dask_util
from cellmap_analyze.util.dask_util import create_block_from_index
from cellmap_analyze.util.grid_util import crop_to_roi, grid_of
from cellmap_analyze.util.io_util import TimingMessager, get_leaf_name_from_path, split_dataset_path
from cellmap_analyze.util.measure_util import calculate_surface_areas_voxelwise
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.zarr_util import create_multiscale_dataset, open_dataset

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Sheetness in [0, 1] is stored as 1 + round(254 * sheetness), 0 outside objects.
LEVELS = 254


def to_level(sheetness):
    return (1 + np.rint(np.clip(sheetness, 0, 1) * LEVELS)).astype(np.uint8)


def from_level(level):
    return (np.asarray(level, dtype=float) - 1) / LEVELS


def _scale_sigmas(voxel_size, scale_steps, steps_per_octave=2):
    """COSEM's scales: sigmas from half the finest voxel, doubling every
    ``steps_per_octave`` steps, in each axis's voxels (one row per scale)."""
    voxel_size = np.asarray(voxel_size, dtype=float)
    target = 0.5 * 2.0 ** (np.arange(scale_steps) / steps_per_octave)
    return target[:, None] * voxel_size.min() / voxel_size[None, :]


def scale_series(voxel_size, scale_steps=12, steps_per_octave=2):
    """COSEM's scale series (``SparkCurvature.sigmaSeries``): the blur to add
    at each step, taking the mask itself as blurred by half a voxel, and the
    sigma each step's derivatives are normalised by."""
    sigmas = _scale_sigmas(voxel_size, scale_steps, steps_per_octave)
    clipped = np.maximum(0.5, sigmas)
    increments = np.zeros_like(clipped)
    increments[1:] = np.sqrt(np.maximum(0, clipped[1:] ** 2 - clipped[:-1] ** 2))
    # COSEM accumulates the sigmas themselves, not the increments, so this is
    # about 1.4 times the blur; its noise term (c = 0.5) is tuned with it.
    normalisation = np.sqrt(np.cumsum(sigmas[:, -1] ** 2))
    return increments, normalisation


def sheetness_halo(voxel_size, scale_steps=12):
    """COSEM's halo for the scale series: the half-kernel of the sum of all its
    sigmas, plus the Hessian's 2 voxels. It also covers the exact Hessian's
    blur and derivative kernels."""
    sigmas = _scale_sigmas(voxel_size, scale_steps)
    max_sigma = np.sqrt((sigmas**2).sum(axis=0))
    return tuple(int(max(2, int(3 * s + 0.5) + 1)) + 2 for s in max_sigma)


# The exact Hessian blurs the mask to all but this sigma (voxels), then takes
# Gaussian derivatives of this sigma at each point; smaller scales are taken
# on the mask itself.
DERIVATIVE_SIGMA = 1.0
# Its noise term follows Frangi's rule: half the largest scale-normalised
# Hessian norm a binary structure reaches (about 0.48, at a sheet's middle when
# sigma is half its thickness).
EXACT_NOISE = 0.25


def _derivative_kernels(sigma):
    """Sampled Gaussian kernels of ``sigma`` and their first and second
    derivatives, out to 4 sigma, for correlating with a patch."""
    radius = int(4 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    phi = np.exp(-0.5 * x**2 / sigma**2)
    phi /= phi.sum()
    # correlating with G'(-x) takes the first derivative
    return radius, (phi, x / sigma**2 * phi, (x**2 - sigma**2) / sigma**4 * phi)


def _gaussian_derivative_hessians(field, points, sigma, batch=8192):
    """The Hessian of ``field`` blurred by ``sigma`` (per axis) at ``points``,
    by Gaussian derivatives over a patch around each one."""
    kernels = [_derivative_kernels(s) for s in sigma]
    offsets = [np.arange(-r, r + 1) for r, _ in kernels]
    orders = ((2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1))
    hessians = np.empty((len(points), 3, 3))
    for b in range(0, len(points), batch):
        p = points[b : b + batch]
        patch = field[
            p[:, 0, None, None, None] + offsets[0][None, :, None, None],
            p[:, 1, None, None, None] + offsets[1][None, None, :, None],
            p[:, 2, None, None, None] + offsets[2][None, None, None, :],
        ]
        along_x = [patch @ kernels[2][1][o] for o in range(3)]
        for oz, oy, ox in orders:
            value = (along_x[ox] @ kernels[1][1][oy]) @ kernels[0][1][oz]
            d, e = [axis for axis, o in enumerate((oz, oy, ox)) for _ in range(o)]
            hessians[b : b + batch, d, e] = value
            hessians[b : b + batch, e, d] = value
    return hessians


def _cosem_hessians(mask, points, voxel_size, scale_steps):
    """Each scale's Hessian as COSEM takes it: central differences of central
    differences of the incrementally blurred mask, which underestimates it
    below a blur of about 2 voxels."""
    increments, normalisation = scale_series(voxel_size, scale_steps)
    smoothed = mask.astype(np.float64)
    unit = np.eye(3, dtype=np.intp)

    def sample(offset):
        p = points + offset
        return smoothed[p[:, 0], p[:, 1], p[:, 2]]

    for increment, norm in zip(increments, normalisation):
        if np.any(increment > 0):
            smoothed = gaussian_filter(smoothed, increment, mode="constant", truncate=3.0)
        hessian = np.empty((len(points), 3, 3))
        for d in range(3):
            for e in range(d, 3):
                h = (
                    sample(unit[d] + unit[e])
                    - sample(unit[d] - unit[e])
                    - sample(unit[e] - unit[d])
                    + sample(-unit[d] - unit[e])
                ) * (norm / 2) ** 2
                hessian[:, d, e] = h
                hessian[:, e, d] = h
        yield hessian


def _exact_hessians(mask, points, voxel_size, scale_steps):
    """Each scale's Hessian by Gaussian derivatives of that scale's sigma,
    scale-normalised by sigma squared in physical units. Scales under a voxel
    are skipped: there any digitised structure looks flat, so thin tubes
    would score as sheets."""
    sigmas = _scale_sigmas(voxel_size, scale_steps)
    sigmas = sigmas[sigmas.max(axis=1) >= 1 - 1e-9]
    smoothed = mask.astype(np.float64)
    blurred = np.zeros(3)
    for sigma in sigmas:
        derivative = np.minimum(sigma, DERIVATIVE_SIGMA)
        pre = np.sqrt(sigma**2 - derivative**2)
        increment = np.sqrt(np.maximum(pre**2 - blurred**2, 0))
        if np.any(increment > 0):
            smoothed = gaussian_filter(smoothed, increment, mode="constant", truncate=4.0)
            blurred = pre
        yield _gaussian_derivative_hessians(smoothed, points, derivative) * np.outer(sigma, sigma)


def hessian_sheetness(mask, points, voxel_size, scale_steps=12, hessian="exact"):
    """Sheetness of the binary ``mask`` at ``points`` (voxel indices at least
    the derivative kernels' reach inside it): over scales, the largest
    Descoteaux et al. (2005) sheet measure of the blurred mask's Hessian whose
    largest eigenvalue is negative,
    ``exp(-Rsheet^2 / 2a^2) (1 - exp(-Rblob^2 / 2b^2)) (1 - exp(-S^2 / 2c^2))``
    with a = b = 0.5, ``Rsheet = |l1 / l2|``,
    ``Rblob = |2 |l2| - |l1| - |l0|| / |l2|`` and S the eigenvalues' norm
    (``|l0| <= |l1| <= |l2|``).

    ``hessian="exact"`` takes Gaussian derivatives (c = 0.25) over the scales
    from a voxel up: sheets of any thickness score about 0.7-0.87, tubes 4 or
    more voxels wide about 0.12-0.16 and thinner ones up to 0.3.
    ``"cosem"`` is COSEM's ``SparkCurvature`` (c = 0.5), whose finite
    differences underestimate curvature below a blur of about 2 voxels:
    sheets 3 or more voxels thick score about 0.7-0.85 but thinner ones
    0.3-0.5, tubes about 0.1-0.2."""
    points = np.asarray(points, dtype=np.intp)
    best = np.zeros(len(points))
    if len(points) == 0:
        return best
    if hessian == "exact":
        hessians, noise = _exact_hessians(mask, points, voxel_size, scale_steps), EXACT_NOISE
    elif hessian == "cosem":
        hessians, noise = _cosem_hessians(mask, points, voxel_size, scale_steps), 0.5
    else:
        raise ValueError(f"hessian must be 'exact' or 'cosem', not {hessian!r}")
    for h in hessians:
        eigenvalues = np.linalg.eigvalsh(h)
        eigenvalues = np.take_along_axis(
            eigenvalues, np.argsort(np.abs(eigenvalues), axis=1), axis=1
        )
        a0, a1, a2 = np.abs(eigenvalues).T
        with np.errstate(divide="ignore", invalid="ignore"):
            r_blob = np.abs(2 * a2 - a1 - a0) / a2
            r_sheet = a1 / a2
            measure = (
                np.exp(-(r_sheet**2) / 0.5)
                * (1 - np.exp(-(r_blob**2) / 0.5))
                * (1 - np.exp(-(a0**2 + a1**2 + a2**2) / (2 * noise**2)))
            )
        better = (eigenvalues[:, 2] <= 0) & (measure > best)
        best[better] = measure[better]
    return best


def _distance_at(labels, points, voxel_size):
    """Each point's distance to the nearest voxel of another label (background
    included), in nm, and whether it is exact (nearer than anything outside
    ``labels``)."""
    distance = edt_module.edt(labels, anisotropy=tuple(voxel_size), black_border=False)
    d = distance[points[:, 0], points[:, 1], points[:, 2]].astype(float)
    shape = np.array(labels.shape)
    outside = np.min(
        np.minimum(points + 1, shape - points) * np.asarray(voxel_size), axis=1
    )
    return d, bool(np.all(d <= outside))


def _worker_memory_bytes():
    """The memory this task's dask worker may use, or outside a worker the
    memory the machine has available."""
    try:
        from distributed import get_worker

        limit = get_worker().memory_manager.memory_limit
        if limit:
            return int(limit)
    except (ImportError, ValueError):
        pass
    import psutil

    return int(psutil.virtual_memory().available)


def _widen(padding, block_shape, bytes_per_voxel, max_padding, problem, memory=None):
    """``padding`` (voxels per axis) doubled, as far as a read of the block
    with it fits in half the worker's ``memory`` (and within ``max_padding``
    voxels, if given). Raises once it can grow no further, since the result
    would not be exact."""
    padding = np.asarray(padding)
    wider = 2 * padding
    if max_padding is not None:
        wider = np.maximum(padding, np.minimum(wider, max_padding))
    memory = _worker_memory_bytes() if memory is None else memory

    def fits(step):
        p = np.minimum(padding + step, wider)
        return float(np.prod(np.array(block_shape) + 2 * p)) * bytes_per_voxel <= memory / 2

    # the largest step towards ``wider`` that fits
    low, high = 0, int(np.max(wider - padding))
    while low < high:
        mid = (low + high + 1) // 2
        low, high = (mid, high) if fits(mid) else (low, mid - 1)
    wider = np.minimum(padding + low, wider)
    if np.array_equal(wider, padding):
        limit = (
            f"max_padding_voxels={max_padding}"
            if max_padding is not None and np.all(2 * padding > max_padding)
            else f"half of the worker's {memory / 1e9:.1f} GB"
        )
        raise RuntimeError(
            f"{problem} The read, now {tuple(int(p) for p in padding)} voxels on each side, "
            f"can't grow within {limit}: run at a coarser resolution, give workers more "
            f"memory or use smaller blocks."
        )
    return wider


def _unsigned_labels(labels):
    if labels.dtype == bool:
        return labels.view(np.uint8)
    if labels.dtype.kind == "i":
        return labels.view(labels.dtype.str.replace("i", "u"))
    return labels


def _group_sums(keys, *values):
    """Unique rows of ``keys`` (an (n, k) integer array) with their counts and
    the sums of each of ``values`` over them."""
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    counts = np.bincount(inverse, minlength=len(unique))
    sums = [np.bincount(inverse, weights=v, minlength=len(unique)) for v in values]
    return unique, counts, sums


class Sheetness(ComputeConfigMixin):
    """COSEM's sheetness and thickness of objects (e.g. ER) from their medial
    surfaces (``TopologicalThinning`` with ``medial_surface``), ported from
    ``SparkCurvature`` and ``SparkCalculatePropertiesFromMedialSurface``.

    1. At each medial-surface voxel: its sheetness (``hessian_sheetness``, on
       its own object's mask, over ``scale_steps`` scales, by Gaussian
       derivatives or with ``hessian="cosem"`` COSEM's finite differences) and
       thickness,
       twice its distance to another label: within half a voxel of the
       voxelised thickness for sheets at any angle, but a full voxel high for
       an axis-aligned sheet an odd number of voxels thick.
    2. Every other voxel of an object takes the mean sheetness of its nearest
       medial-surface voxels, by geodesic distance through the object (COSEM's
       2023 method; its 2021 one spread sheetness over medial balls, which
       approximates the object, missing some surface voxels and taking in a
       few background ones).

    Both steps widen a block's read until its distances are exact, as far as
    the read fits in half its worker's memory (and ``max_padding_voxels`` on
    each side, if given); an object too thick for that, or one with no medial
    surface, stops the run with an error.

    Writes ``output_path``, the sheetness of every object voxel, stored as
    ``1 + round(254 * sheetness)`` with 0 outside objects,
    ``{output_path}_at_medial_surface`` and ``{output_path}_thickness``
    (float32, nm) on the medial surface, and to ``csv_path`` per object:

    - ``{name}.csv``: volume, surface area, volume- and surface-weighted mean
      sheetness, medial-surface voxels and thickness;
    - ``{name}_histograms.csv``: volume and surface area by sheetness
      (COSEM's ``sheetnessVolumeAndAreaHistograms``);
    - ``{name}_vs_thickness.csv``: medial-surface voxels by sheetness and
      thickness bin (COSEM's ``sheetnessVsThicknessHistogram``).

    Surface area counts voxel faces against other labels, as ``Measure``
    does. The scales are in voxels, so compare datasets at the same
    resolution.
    """

    def __init__(
        self,
        segmentation_path,
        medial_surface_path,
        output_path,
        csv_path,
        scale_steps=12,
        hessian="exact",
        thickness_bin_nm=8,
        max_padding_voxels=None,
        block_shape_voxels=None,
        block_multiplier=4,
        roi=None,
        num_workers=10,
        chunk_shape=None,
    ):
        super().__init__(num_workers)
        self.segmentation_path = segmentation_path
        self.segmentation_vol = open_dataset(segmentation_path)
        self.medial_surface_vol = open_dataset(medial_surface_path)
        self.output_path = str(output_path).rstrip("/")
        os.makedirs(split_dataset_path(self.output_path)[0], exist_ok=True)
        self.csv_path = str(csv_path).rstrip("/")
        self.name = get_leaf_name_from_path(self.output_path) or "sheetness"

        self.grid = crop_to_roi(grid_of(self.segmentation_vol, chunk_shape), roi)
        chunks = np.array(self.grid.chunks)
        if block_shape_voxels is not None:
            self.block_shape = tuple(int(b) for b in block_shape_voxels)
        else:
            self.block_shape = tuple(int(c) * int(block_multiplier) for c in chunks)
        if np.any(np.array(self.block_shape) % chunks):
            raise ValueError(
                f"block shape {self.block_shape} must be a multiple of the chunks {tuple(chunks)}"
            )
        self.num_blocks = dask_util.get_num_blocks(self.grid, self.block_shape)
        self.block_grid_shape = tuple(
            int(np.ceil(s / b)) for s, b in zip(self.grid.shape, self.block_shape)
        )

        self.voxel_size = np.array(self.grid.voxel_size, dtype=float)
        self.scale_steps = int(scale_steps)
        if hessian not in ("exact", "cosem"):
            raise ValueError(f"hessian must be 'exact' or 'cosem', not {hessian!r}")
        self.hessian = hessian
        self.halo = sheetness_halo(self.voxel_size, self.scale_steps)
        self.thickness_bin_nm = float(thickness_bin_nm)
        self.max_padding_voxels = None if max_padding_voxels is None else int(max_padding_voxels)

        self.sheetness_vol = create_multiscale_dataset(
            self.output_path, dtype=np.uint8, grid=self.grid
        )
        self.at_medial_surface_vol = create_multiscale_dataset(
            f"{self.output_path}_at_medial_surface", dtype=np.float32, grid=self.grid
        )
        self.thickness_vol = create_multiscale_dataset(
            f"{self.output_path}_thickness", dtype=np.float32, grid=self.grid
        )

    @staticmethod
    def medial_surface_block(
        block_index,
        grid,
        block_shape,
        halo,
        segmentation_vol,
        medial_surface_vol,
        at_medial_surface_vol,
        thickness_vol,
        scale_steps,
        hessian,
        thickness_bin_nm,
        max_padding,
    ):
        """Sheetness and thickness at a block's medial-surface voxels."""
        block = create_block_from_index(grid, block_index, padding=halo, block_shape=block_shape)
        medial = medial_surface_vol.read_onto(grid, block.begin, block.end)
        local = np.argwhere(medial > 0)
        if len(local) == 0:
            # its object voxels, if any, still cost the next step
            voxels = int(np.count_nonzero(segmentation_vol.read_onto(grid, block.begin, block.end)))
            if voxels == 0:
                return None
            return {"index": block_index, "voxels": voxels, "max_radius_nm": 0.0, "vs_thickness": None}
        ids = medial[tuple(local.T)].astype(np.uint64)
        voxel_size = np.array(grid.voxel_size, dtype=float)
        labels = segmentation_vol.read_onto(grid, block.read_begin, block.read_end)
        points = local + (np.array(block.begin) - np.array(block.read_begin))

        sheetness = np.zeros(len(points))
        halo = np.array(halo)
        for id_value in np.unique(ids):
            mine = ids == id_value
            lo = points[mine].min(axis=0) - halo
            hi = points[mine].max(axis=0) + halo + 1
            box = tuple(slice(a, b) for a, b in zip(lo, hi))
            sheetness[mine] = hessian_sheetness(
                labels[box] == id_value, points[mine] - lo, voxel_size, scale_steps, hessian
            )

        distance, exact = _distance_at(labels, points, voxel_size)
        padding = halo
        while not exact:
            # an object thicker than the halo: widen the read until the
            # distances are exact
            # the labels and their distance transform
            padding = _widen(
                padding, block_shape, labels.dtype.itemsize + 8, max_padding,
                f"Block {block_index}: a medial-surface voxel is farther from its object's"
                " edge than the read reaches.",
            )
            wide = create_block_from_index(grid, block_index, padding=padding, block_shape=block_shape)
            wide_labels = segmentation_vol.read_onto(grid, wide.read_begin, wide.read_end)
            wide_points = local + (np.array(wide.begin) - np.array(wide.read_begin))
            distance, exact = _distance_at(wide_labels, wide_points, voxel_size)
        thickness = 2 * distance

        for vol, per_voxel in ((at_medial_surface_vol, sheetness), (thickness_vol, thickness)):
            values = np.zeros(medial.shape, dtype=np.float32)
            values[tuple(local.T)] = per_voxel
            vol.write_index(block.begin, block.end, values)

        keys = np.stack(
            [ids, to_level(sheetness), np.floor(thickness / thickness_bin_nm)], axis=1
        ).astype(np.int64)
        unique, counts, (thickness_sum, thickness_sq) = _group_sums(
            keys, thickness, thickness**2
        )
        return {
            "index": block_index,
            "voxels": int(np.count_nonzero(labels[block.inner])),
            "max_radius_nm": float(distance.max()),
            "vs_thickness": pd.DataFrame(
                {
                    "id": unique[:, 0],
                    "level": unique[:, 1],
                    "thickness_bin": unique[:, 2],
                    "count": counts,
                    "thickness_sum": thickness_sum,
                    "thickness_sq": thickness_sq,
                }
            ),
        }

    @staticmethod
    def volume_block(
        block_index,
        grid,
        block_shape,
        paddings,
        segmentation_vol,
        medial_surface_vol,
        at_medial_surface_vol,
        sheetness_vol,
        max_padding,
    ):
        """Sheetness of a block's voxels from their nearest medial-surface
        voxels, and its volume and surface area by sheetness."""
        block = create_block_from_index(grid, block_index, block_shape=block_shape)
        begin, end = np.array(block.begin), np.array(block.end)
        shelled = segmentation_vol.read_onto(grid, begin - 1, end + 1)
        labels = shelled[1:-1, 1:-1, 1:-1]
        inside = labels > 0
        if not inside.any():
            return None
        voxel_size = np.array(grid.voxel_size, dtype=float)

        # Widen the read until every voxel's nearest medial-surface voxel is
        # nearer than anything outside it.
        padding = np.array(paddings[block_index])
        if max_padding is not None:
            padding = np.minimum(padding, max_padding)
        while True:
            padded = create_block_from_index(grid, block_index, padding=padding, block_shape=block_shape)
            padded_labels = np.ascontiguousarray(
                _unsigned_labels(segmentation_vol.read_onto(grid, padded.read_begin, padded.read_end))
            )
            medial = medial_surface_vol.read_onto(grid, padded.read_begin, padded.read_end)
            sources = np.argwhere(medial > 0)
            values = at_medial_surface_vol.read_onto(grid, padded.read_begin, padded.read_end)[
                tuple(sources.T)
            ]
            distance, sheetness = nearest_source_values(padded_labels, sources, values, tuple(voxel_size))
            distance, sheetness = distance[padded.inner], sheetness[padded.inner]
            if np.all(distance[inside] <= np.min(padding * voxel_size)):
                break
            # the labels and medial surface, the values read on it, and the
            # geodesic search's distances, sums, counts and output
            padding = _widen(
                padding, block_shape, 2 * padded_labels.dtype.itemsize + 40, max_padding,
                f"Block {block_index}: an object voxel's nearest medial-surface voxel, by"
                " paths through its object, is farther than the read reaches, or its object"
                " has none (was the medial surface thinned from this segmentation?).",
            )

        level = np.where(inside, to_level(np.nan_to_num(sheetness)), 0).astype(np.uint8)
        sheetness_vol.write_index(block.begin, block.end, level)

        face_areas = (
            voxel_size[1] * voxel_size[2],
            voxel_size[0] * voxel_size[2],
            voxel_size[0] * voxel_size[1],
        )
        areas = calculate_surface_areas_voxelwise(
            shelled, face_areas, do_zero_padding=False
        )[1:-1, 1:-1, 1:-1]
        ids = labels[inside].astype(np.int64)
        levels = level[inside].astype(np.int64)
        area = areas[inside]
        value = sheetness[inside]
        unique, counts, (area_sum, value_sum, value_area_sum) = _group_sums(
            np.stack([ids, levels], axis=1), area, value, value * area
        )
        return {
            "index": block_index,
            "histograms": pd.DataFrame(
                {
                    "id": unique[:, 0],
                    "level": unique[:, 1],
                    "voxels": counts,
                    "area": area_sum,
                    "sheetness_sum": value_sum,
                    "sheetness_area_sum": value_area_sum,
                }
            ),
        }

    @staticmethod
    def _partition(block_indices, fn, *args):
        results = [fn(i, *args) for i in block_indices]
        return [r for r in results if r is not None]

    def _run(self, msg, fn, *args, costs=None):
        partitions = dask_util.block_partitions(self.num_blocks, self.num_workers, costs)
        with TimingMessager(msg, logger):
            return dask_util.compute_partitions(
                partitions, Sheetness._partition, fn, *args, compute_args=self.compute_args
            )

    def _initial_paddings(self, max_radius_nm):
        """Per block, a first guess at how far its voxels' nearest
        medial-surface voxels can be: half again the largest medial radius in
        blocks within reach of it."""
        radius = np.zeros(self.num_blocks)
        for index, r in max_radius_nm.items():
            radius[index] = r
        radius = radius.reshape(self.block_grid_shape)
        reach = int(np.ceil(1.5 * radius.max() / np.min(np.array(self.block_shape) * self.voxel_size)))
        if reach > 0:
            radius = maximum_filter(radius, size=2 * reach + 1, mode="constant", cval=0)
        return [
            tuple(int(np.ceil(1.5 * r / v)) + 2 for v in self.voxel_size)
            for r in radius.ravel()
        ]

    def calculate(self):
        with dask_util.start_dask(self.num_workers, "sheetness", logger):
            medial_results = self._run(
                "Calculating sheetness and thickness on the medial surface",
                Sheetness.medial_surface_block,
                self.grid,
                self.block_shape,
                self.halo,
                self.segmentation_vol,
                self.medial_surface_vol,
                self.at_medial_surface_vol,
                self.thickness_vol,
                self.scale_steps,
                self.hessian,
                self.thickness_bin_nm,
                self.max_padding_voxels,
            )
            paddings = self._initial_paddings(
                {r["index"]: r["max_radius_nm"] for r in medial_results}
            )
            # a block's geodesic assignment takes time in proportion to its
            # object voxels, so start the densest blocks first
            voxels = np.zeros(self.num_blocks)
            for r in medial_results:
                voxels[r["index"]] = r["voxels"]
            volume_results = self._run(
                "Assigning sheetness from the nearest medial-surface voxels",
                Sheetness.volume_block,
                self.grid,
                self.block_shape,
                paddings,
                self.segmentation_vol,
                self.medial_surface_vol,
                self.at_medial_surface_vol,
                self.sheetness_vol,
                self.max_padding_voxels,
                costs=voxels,
            )
        self._write_csvs(medial_results, volume_results)

    def _write_csvs(self, medial_results, volume_results):
        os.makedirs(self.csv_path, exist_ok=True)
        voxel_volume = float(np.prod(self.voxel_size))
        empty = pd.DataFrame(columns=["id", "level"])

        frames = [r["vs_thickness"] for r in medial_results if r["vs_thickness"] is not None]
        vs_thickness = (
            pd.concat(frames).groupby(["id", "level", "thickness_bin"], as_index=False).sum()
            if frames
            else empty.assign(thickness_bin=[], count=[], thickness_sum=[], thickness_sq=[])
        )
        frames = [r["histograms"] for r in volume_results]
        histograms = (
            pd.concat(frames).groupby(["id", "level"], as_index=False).sum()
            if frames
            else empty.assign(voxels=[], area=[], sheetness_sum=[], sheetness_area_sum=[])
        )

        per_object = histograms.groupby("id").agg(
            voxels=("voxels", "sum"),
            area=("area", "sum"),
            sheetness_sum=("sheetness_sum", "sum"),
            sheetness_area_sum=("sheetness_area_sum", "sum"),
        )
        medial = vs_thickness.groupby("id").agg(
            count=("count", "sum"),
            thickness_sum=("thickness_sum", "sum"),
            thickness_sq=("thickness_sq", "sum"),
        ).reindex(per_object.index, fill_value=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_thickness = medial["thickness_sum"] / medial["count"]
            summary = pd.DataFrame(
                {
                    "Volume (nm^3)": per_object["voxels"] * voxel_volume,
                    "Surface Area (nm^2)": per_object["area"],
                    "Mean Sheetness": per_object["sheetness_sum"] / per_object["voxels"],
                    "Mean Surface Sheetness": per_object["sheetness_area_sum"] / per_object["area"],
                    "Medial Surface Voxels": medial["count"].astype(np.int64),
                    "Mean Thickness (nm)": mean_thickness,
                    "Thickness Std (nm)": np.sqrt(
                        np.maximum(medial["thickness_sq"] / medial["count"] - mean_thickness**2, 0)
                    ),
                }
            )
        summary.index.name = "Object ID"
        summary.to_csv(f"{self.csv_path}/{self.name}.csv")

        pd.DataFrame(
            {
                "Object ID": histograms["id"].to_numpy(),
                "Sheetness": from_level(histograms["level"]),
                "Volume (nm^3)": histograms["voxels"].to_numpy() * voxel_volume,
                "Surface Area (nm^2)": histograms["area"].to_numpy(),
            }
        ).to_csv(f"{self.csv_path}/{self.name}_histograms.csv", index=False)

        pd.DataFrame(
            {
                "Object ID": vs_thickness["id"].to_numpy(),
                "Sheetness": from_level(vs_thickness["level"]),
                "Thickness Bin Start (nm)": vs_thickness["thickness_bin"].to_numpy() * self.thickness_bin_nm,
                "Medial Surface Voxels": vs_thickness["count"].to_numpy().astype(np.int64),
            }
        ).to_csv(f"{self.csv_path}/{self.name}_vs_thickness.csv", index=False)

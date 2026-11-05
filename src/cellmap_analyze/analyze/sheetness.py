import logging
import os

import dask.bag as db
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


def scale_series(voxel_size, scale_steps=12, steps_per_octave=2):
    """COSEM's scales (``SparkCurvature.sigmaSeries``): sigmas from half the
    finest voxel, doubling every ``steps_per_octave`` steps, in each axis's
    voxels. Returns the blur to add at each step (taking the mask itself as
    blurred by half a voxel) and the sigma each step's derivatives are
    normalised by."""
    voxel_size = np.asarray(voxel_size, dtype=float)
    target = 0.5 * 2.0 ** (np.arange(scale_steps) / steps_per_octave)
    sigmas = target[:, None] * voxel_size.min() / voxel_size[None, :]
    clipped = np.maximum(0.5, sigmas)
    increments = np.zeros_like(clipped)
    increments[1:] = np.sqrt(np.maximum(0, clipped[1:] ** 2 - clipped[:-1] ** 2))
    # COSEM accumulates the sigmas themselves, not the increments, so this is
    # about 1.4 times the blur; its noise term (c = 0.5) is tuned with it.
    normalisation = np.sqrt(np.cumsum(sigmas[:, -1] ** 2))
    return increments, normalisation


def sheetness_halo(voxel_size, scale_steps=12):
    """COSEM's halo for the scale series: the half-kernel of the sum of all its
    sigmas, plus the Hessian's 2 voxels."""
    voxel_size = np.asarray(voxel_size, dtype=float)
    target = 0.5 * 2.0 ** (np.arange(scale_steps) / 2)
    sigmas = target[:, None] * voxel_size.min() / voxel_size[None, :]
    max_sigma = np.sqrt((sigmas**2).sum(axis=0))
    return tuple(int(max(2, int(3 * s + 0.5) + 1)) + 2 for s in max_sigma)


def hessian_sheetness(mask, points, increments, normalisation):
    """COSEM's sheetness (``SparkCurvature``) of the binary ``mask`` at
    ``points`` (voxel indices at least 2 voxels inside it): over scales, the
    largest Frangi sheet measure of the blurred mask's Hessian whose largest
    eigenvalue is negative,
    ``exp(-Rsheet^2 / 2a^2) (1 - exp(-Rblob^2 / 2b^2)) (1 - exp(-S^2 / 2c^2))``
    with a = b = c = 0.5, ``Rsheet = |l1 / l2|``,
    ``Rblob = |2 |l2| - |l1| - |l0|| / |l2|`` and S the eigenvalues' norm
    (``|l0| <= |l1| <= |l2|``). Sheets score about 0.7-0.85 once 3 or more
    voxels thick, tubes about 0.1-0.2."""
    points = np.asarray(points, dtype=np.intp)
    best = np.zeros(len(points))
    if len(points) == 0:
        return best
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
                # central differences of central differences, as COSEM takes them
                h = (
                    sample(unit[d] + unit[e])
                    - sample(unit[d] - unit[e])
                    - sample(unit[e] - unit[d])
                    + sample(-unit[d] - unit[e])
                ) * (norm / 2) ** 2
                hessian[:, d, e] = h
                hessian[:, e, d] = h
        eigenvalues = np.linalg.eigvalsh(hessian)
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
                * (1 - np.exp(-(a0**2 + a1**2 + a2**2) / 0.5))
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
       its own object's mask, over ``scale_steps`` scales) and thickness,
       twice its distance to another label: within half a voxel of the
       voxelised thickness for sheets at any angle, but a full voxel high for
       an axis-aligned sheet an odd number of voxels thick.
    2. Every other voxel of an object takes the mean sheetness of its nearest
       medial-surface voxels, by geodesic distance through the object (COSEM's
       2023 method; its 2021 one spread sheetness over medial balls, which
       left the object's surface layer uncovered or, unmasked as COSEM had
       it, covered a ring of background instead).

    Writes ``output_path``, the sheetness of every object voxel, stored as
    ``1 + round(254 * sheetness)`` with 0 outside objects (or where no
    medial-surface voxel of the object is reachable),
    ``{output_path}_at_medial_surface`` and ``{output_path}_thickness``
    (float32, nm) on the medial surface, and to ``csv_path`` per object:

    - ``{name}.csv``: volume, surface area, volume- and surface-weighted mean
      sheetness, unreached volume, medial-surface voxels and thickness;
    - ``{name}_histograms.csv``: volume and surface area by sheetness
      (COSEM's ``sheetnessVolumeAndAreaHistograms``);
    - ``{name}_vs_thickness.csv``: medial-surface voxels by sheetness and
      thickness bin (COSEM's ``sheetnessVsThicknessHistogram``).

    Surface area counts voxel faces against other labels, as ``Measure``
    does. The scales are in voxels, and sheets thinner than 3 voxels score
    lower (about 0.3-0.5), so compare datasets at the same resolution.
    """

    def __init__(
        self,
        segmentation_path,
        medial_surface_path,
        output_path,
        csv_path,
        scale_steps=12,
        thickness_bin_nm=8,
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
        self.increments, self.normalisation = scale_series(self.voxel_size, self.scale_steps)
        self.halo = sheetness_halo(self.voxel_size, self.scale_steps)
        self.thickness_bin_nm = float(thickness_bin_nm)

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
        increments,
        normalisation,
        thickness_bin_nm,
    ):
        """Sheetness and thickness at a block's medial-surface voxels."""
        block = create_block_from_index(grid, block_index, padding=halo, block_shape=block_shape)
        medial = medial_surface_vol.read_onto(grid, block.begin, block.end)
        local = np.argwhere(medial > 0)
        if len(local) == 0:
            return None
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
                labels[box] == id_value, points[mine] - lo, increments, normalisation
            )

        distance, exact = _distance_at(labels, points, voxel_size)
        padding = halo
        while not exact:
            # an object thicker than the halo: widen the read until the
            # distances are exact
            padding = padding * 2
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
        # nearer than anything outside it; a voxel still unreached after that
        # has no medial surface in its object nearby.
        padding = np.array(paddings[block_index])
        for attempt in range(3):
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
            padding = padding * 2

        reached = inside & np.isfinite(distance)
        level = np.where(reached, to_level(np.nan_to_num(sheetness)), 0).astype(np.uint8)
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
        value = np.nan_to_num(sheetness[inside])
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

    def _run(self, msg, fn, *args):
        npartitions = dask_util.guesstimate_npartitions(self.num_blocks, self.num_workers)
        bag = db.range(self.num_blocks, npartitions=max(1, min(npartitions, self.num_blocks)))
        with TimingMessager(msg, logger):
            return bag.map_partitions(Sheetness._partition, fn, *args).compute(
                **self.compute_args
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
                self.increments,
                self.normalisation,
                self.thickness_bin_nm,
            )
            paddings = self._initial_paddings(
                {r["index"]: r["max_radius_nm"] for r in medial_results}
            )
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
            )
        self._write_csvs(medial_results, volume_results)

    def _write_csvs(self, medial_results, volume_results):
        os.makedirs(self.csv_path, exist_ok=True)
        voxel_volume = float(np.prod(self.voxel_size))
        empty = pd.DataFrame(columns=["id", "level"])

        frames = [r["vs_thickness"] for r in medial_results]
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
            voxels=("voxels", "sum"), area=("area", "sum")
        )
        covered = histograms[histograms["level"] > 0].groupby("id").agg(
            voxels=("voxels", "sum"),
            area=("area", "sum"),
            sheetness_sum=("sheetness_sum", "sum"),
            sheetness_area_sum=("sheetness_area_sum", "sum"),
        ).reindex(per_object.index, fill_value=0)
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
                    "Mean Sheetness": covered["sheetness_sum"] / covered["voxels"],
                    "Mean Surface Sheetness": covered["sheetness_area_sum"] / covered["area"],
                    "Unreached Volume (nm^3)": (per_object["voxels"] - covered["voxels"]) * voxel_volume,
                    "Medial Surface Voxels": medial["count"].astype(np.int64),
                    "Mean Thickness (nm)": mean_thickness,
                    "Thickness Std (nm)": np.sqrt(
                        np.maximum(medial["thickness_sq"] / medial["count"] - mean_thickness**2, 0)
                    ),
                }
            )
        summary.index.name = "Object ID"
        summary.to_csv(f"{self.csv_path}/{self.name}.csv")

        histograms = histograms[histograms["level"] > 0]
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

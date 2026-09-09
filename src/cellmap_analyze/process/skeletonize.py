import numpy as np
import pandas as pd
import networkx as nx
import edt as edt_module
from cellmap_analyze.util import dask_util
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.skeleton_util import (
    CustomSkeleton,
    skimage_to_custom_skeleton_fast,
)
from scipy.ndimage import zoom
from skimage.morphology import skeletonize
from tqdm import tqdm
import logging
import os
import uuid
import json

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- Unbridged adjacency removal helpers ---
#
# Two voxels can be adjacent in three ways:
#   face-adjacent   (6-conn):  differ in 1 axis, share a face
#   edge-adjacent   (18-conn): differ in 2 axes, share an edge
#   vertex-adjacent (26-conn): differ in 3 axes, share only a corner point
#
# A "bridge" is a third voxel that is face-adjacent (or edge-adjacent,
# depending on mode) to BOTH voxels in a pair. If at least one bridge
# voxel is foreground, the pair is connected at the desired connectivity
# without relying on the weaker adjacency.
#
# Only half the directions are listed (first nonzero component positive)
# because the A->B and B->A relationships are symmetric.

# Edge-adjacent pairs and their face bridges.
# For an offset like (1,1,0), the 2 face bridges are found by zeroing
# each nonzero component: (1,0,0) and (0,1,0).
_EDGE_ADJ_FACE_BRIDGES = [
    ((1, 1, 0), [(1, 0, 0), (0, 1, 0)]),
    ((1, -1, 0), [(1, 0, 0), (0, -1, 0)]),
    ((1, 0, 1), [(1, 0, 0), (0, 0, 1)]),
    ((1, 0, -1), [(1, 0, 0), (0, 0, -1)]),
    ((0, 1, 1), [(0, 1, 0), (0, 0, 1)]),
    ((0, 1, -1), [(0, 1, 0), (0, 0, -1)]),
]

# Vertex-adjacent pairs and their face bridges only.
# For an offset like (1,1,1), the 3 face bridges are found by keeping
# one nonzero component at a time: (1,0,0), (0,1,0), (0,0,1).
_VERTEX_ADJ_FACE_BRIDGES = [
    ((1, 1, 1), [(1, 0, 0), (0, 1, 0), (0, 0, 1)]),
    ((1, 1, -1), [(1, 0, 0), (0, 1, 0), (0, 0, -1)]),
    ((1, -1, 1), [(1, 0, 0), (0, -1, 0), (0, 0, 1)]),
    ((1, -1, -1), [(1, 0, 0), (0, -1, 0), (0, 0, -1)]),
]

# Vertex-adjacent pairs with all bridges (3 face + 3 edge = 6 per pair).
# Edge bridges are found by keeping two nonzero components at a time:
# e.g. for (1,1,1) -> (1,1,0), (1,0,1), (0,1,1).
_VERTEX_ADJ_ALL_BRIDGES = [
    (
        (1, 1, 1),
        [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
    ),
    (
        (1, 1, -1),
        [(1, 0, 0), (0, 1, 0), (0, 0, -1), (1, 1, 0), (1, 0, -1), (0, 1, -1)],
    ),
    (
        (1, -1, 1),
        [(1, 0, 0), (0, -1, 0), (0, 0, 1), (1, -1, 0), (1, 0, 1), (0, -1, 1)],
    ),
    (
        (1, -1, -1),
        [(1, 0, 0), (0, -1, 0), (0, 0, -1), (1, -1, 0), (1, 0, -1), (0, -1, -1)],
    ),
]


def _shift(data, dz, dy, dx):
    """Shift a 3D boolean array by (dz, dy, dx), filling vacated edges with False."""
    result = np.zeros_like(data)
    sz_src, sz_dst = _shift_slices(dz, data.shape[0])
    sy_src, sy_dst = _shift_slices(dy, data.shape[1])
    sx_src, sx_dst = _shift_slices(dx, data.shape[2])
    result[sz_dst, sy_dst, sx_dst] = data[sz_src, sy_src, sx_src]
    return result


def _shift_slices(delta, size):
    """Return (source_slice, dest_slice) for a shift of `delta` along an axis of `size`."""
    if delta > 0:
        return slice(0, size - delta), slice(delta, size)
    elif delta < 0:
        return slice(-delta, size), slice(0, size + delta)
    else:
        return slice(None), slice(None)


def remove_unbridged_adjacencies(data, connectivity=6):
    """Remove foreground voxels whose only connection to a neighbor is weaker
    than the specified connectivity.

    For each pair of edge-adjacent or vertex-adjacent foreground voxels,
    checks whether they share a bridging voxel at the desired connectivity
    level. If not, the voxel is marked for removal.

    Uses a Cython implementation when available (much faster for large arrays),
    falling back to a numpy vectorized version otherwise.

    Args:
        data: 3D boolean array.
        connectivity: 6 — keep only face-adjacent connections; remove voxels
                          that are only edge- or vertex-adjacent without a
                          face bridge.
                      18 — keep face- and edge-adjacent connections; remove
                           voxels that are only vertex-adjacent without a
                           face or edge bridge.

    Returns:
        Modified boolean array with unbridged voxels removed.
    """
    if connectivity == 6:
        pairs = _EDGE_ADJ_FACE_BRIDGES + _VERTEX_ADJ_FACE_BRIDGES
    elif connectivity == 18:
        pairs = _VERTEX_ADJ_ALL_BRIDGES
    else:
        raise ValueError(f"connectivity must be 6 or 18, got {connectivity}")

    to_remove = np.zeros_like(data)
    for offset, bridges in pairs:
        neighbor = _shift(data, *offset)
        any_bridge = np.zeros_like(data)
        for b in bridges:
            any_bridge |= _shift(data, *b)
        problem = data & neighbor & ~any_bridge
        to_remove |= problem

    return data & ~to_remove


# Standard binary morphology applied with a structuring element + iterations.
_STANDARD_MORPHOLOGY_OPS = ("erosion", "dilation", "opening", "closing")

# Valid structuring-element connectivity ranks, matching the codebase's
# convention everywhere else (1=faces/6-neighbour, 2=faces+edges/18,
# 3=faces+edges+corners/26).
_STRUCTURE_RANKS = (1, 2, 3)

# Targeted removal of diagonal-only ("corner-touching") connections, which
# otherwise produce spurious skeleton branches. The number names the
# connectivity that is *enforced* (what counts as connected), which is the
# 6/18 convention remove_unbridged_adjacencies accepts.
_CORNER_BRIDGE_OPS = {
    "remove_corner_bridges_6": 6,
    "remove_corner_bridges_18": 18,
    "6": 6,
    "18": 18,
}


def _binary_structure(connectivity):
    from scipy.ndimage import generate_binary_structure

    if connectivity not in _STRUCTURE_RANKS:
        raise ValueError(
            f"connectivity must be one of {list(_STRUCTURE_RANKS)} "
            f"(1=faces, 2=faces+edges, 3=faces+edges+corners); "
            f"got {connectivity!r}"
        )
    return generate_binary_structure(3, connectivity)


def normalize_morphological_operations(operations):
    """Normalize a morphological-operations spec into a list of
    ``{operation, iterations, connectivity}`` dicts.

    Accepts a single op or a list, where each op is either:
    - a string: ``"erosion"``, ``"dilation"``, ``"opening"``, ``"closing"``,
      or a corner-bridge removal (``"remove_corner_bridges_6"``/``"6"``,
      ``"remove_corner_bridges_18"``/``"18"``); or
    - a dict ``{"operation": ..., "iterations": int, "connectivity": 1|2|3}``
      (``iterations``/``connectivity`` optional; default 1 and 1). The
      ``connectivity`` rank matches the rest of the codebase: 1=faces (6),
      2=faces+edges (18), 3=faces+edges+corners (26).
    """
    if operations is None:
        return []
    if isinstance(operations, (str, dict)):
        operations = [operations]

    normalized = []
    for op in operations:
        if isinstance(op, str):
            op = {"operation": op}
        elif not isinstance(op, dict):
            raise TypeError(
                f"each morphological operation must be a str or dict; got "
                f"{type(op).__name__}"
            )
        name = op.get("operation")
        if name in _CORNER_BRIDGE_OPS:
            # Connectivity is fixed by the op name; iterations don't apply.
            normalized.append(
                {
                    "operation": "remove_corner_bridges",
                    "connectivity": _CORNER_BRIDGE_OPS[name],
                    "iterations": 1,
                }
            )
            continue
        if name not in _STANDARD_MORPHOLOGY_OPS:
            raise ValueError(
                f"unknown morphological operation {name!r}; valid: "
                f"{list(_STANDARD_MORPHOLOGY_OPS) + list(_CORNER_BRIDGE_OPS)}"
            )
        iterations = int(op.get("iterations", 1))
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        connectivity = int(op.get("connectivity", 1))
        _binary_structure(connectivity)  # validate connectivity early
        normalized.append(
            {
                "operation": name,
                "iterations": iterations,
                "connectivity": connectivity,
            }
        )
    return normalized


def apply_morphological_operations(data, operations):
    """Apply a sequence of normalized morphological operations to a boolean
    mask, in order, returning the resulting boolean mask."""
    from scipy.ndimage import (
        binary_closing,
        binary_dilation,
        binary_erosion,
        binary_opening,
    )

    funcs = {
        "erosion": binary_erosion,
        "dilation": binary_dilation,
        "opening": binary_opening,
        "closing": binary_closing,
    }
    for op in operations:
        name = op["operation"]
        if name == "remove_corner_bridges":
            data = remove_unbridged_adjacencies(
                data, connectivity=op["connectivity"]
            )
        else:
            data = funcs[name](
                data,
                structure=_binary_structure(op["connectivity"]),
                iterations=op["iterations"],
            )
    return data


def _erosion_to_operations(erosion):
    """Map the legacy ``erosion`` argument to a morphological-operations spec.

    ``True``/``"full"`` -> a single 6-connectivity erosion; ``6``/``18`` ->
    the corresponding corner-bridge removal; ``False``/``None`` -> no ops.
    """
    if erosion is True or erosion == "full":
        return ["erosion"]
    if erosion is False or erosion is None:
        return []
    if erosion in (6, 18, "6", "18"):
        return [str(erosion)]
    raise ValueError(
        f"erosion must be True, False, None, 'full', 6, 18, '6', or '18', "
        f"got {erosion!r}"
    )


class Skeletonize(ComputeConfigMixin):
    def __init__(
        self,
        segmentation_path,
        output_path,
        csv_path=None,
        erosion=True,
        morphological_operations=None,
        min_branch_length_nm=100,
        tolerance_nm=50,
        num_workers=10,
        timeout=5,
        concurrency_limit=None,
        sharded=True,
        shard_bits=1,
        minishard_bits=6,
        retry_on_oom=True,
        memory_retry_max=3,
        peak_bytes_baseline=250_000_000,
        peak_bytes_per_voxel=6.63,
        memory_safety_multiplier=2.0,
        memory_fraction=0.60,
        skeleton_properties=True,
        write_vertex_radius=False,
        prune_only=False,
    ):
        """
        Skeletonize a segmentation, parallelized over IDs.

        Args:
            segmentation_path: Path to the segmentation zarr dataset
            output_path: Path to the output directory for skeletons
            csv_path: Optional path to a CSV with per-object bounding boxes
                     (the kind Measure produces: ``Object ID`` index plus
                     ``MIN X (nm)`` ... ``MAX Z (nm)`` columns). If left as
                     ``None`` (the default), Measure is run on the
                     segmentation to generate one at
                     ``<output_path>/bboxes/<leaf>.csv``. Pass an existing
                     path to skip the auto-measure step. A path that does
                     not exist raises ``FileNotFoundError`` rather than
                     silently auto-generating to that exact location.
            erosion: Legacy shorthand for pre-skeletonization morphology
                     (used only when ``morphological_operations`` is None).
                     True or "full": one 6-connectivity binary erosion.
                     6: targeted removal of edge/vertex-only bridges (keep face-connected).
                     18: targeted removal of vertex-only bridges (keep face+edge-connected).
                     False or None: no operation.
            morphological_operations: Sequence of morphological operations
                     applied (in order) to each object's binary mask before
                     skeletonization. Supersedes ``erosion`` when provided.
                     Each item is a string or a dict:
                       - ``"erosion"`` / ``"dilation"`` / ``"opening"`` /
                         ``"closing"`` -- standard binary morphology;
                       - ``"6"`` / ``"18"`` (a.k.a.
                         ``"remove_corner_bridges_6"`` /
                         ``"remove_corner_bridges_18"``) -- remove diagonal-only
                         connections that spawn spurious branches;
                       - ``{"operation": <name>, "iterations": <int>,
                         "connectivity": 1|2|3}`` for control over the
                         structuring element (rank: 1=faces, 2=+edges,
                         3=+corners; default 1, matching the legacy erosion)
                         and repeat count.
                     E.g. ``["closing", "6"]`` fills small holes then strips
                     corner bridges; ``[{"operation": "opening",
                     "iterations": 2}]`` removes thin protrusions. Radii are
                     measured on the union of the original and processed masks,
                     so shrinking ops (erosion/opening) keep true-object radii
                     while growing ops (dilation/closing) reflect the
                     grown/filled structure.
            min_branch_length_nm: Minimum branch length for pruning (in nm)
            tolerance_nm: Tolerance for simplification (in nm)
            num_workers: Number of parallel workers
            timeout: Timeout for ImageDataInterface reads
            concurrency_limit: tensorstore concurrency limit for
                ``segmentation_idi`` reads. None (default) auto-resolves via
                ``dask_util.resolve_concurrency_limit`` (see
                ``SplitNarrowBridges`` for the full rationale): safe ``1``
                when ``num_workers > 1`` (sibling worker processes may share
                a node/job's CPU allocation), or every CPU actually
                available to this process when ``num_workers <= 1`` (no
                cluster, no siblings). For real lsf/slurm/sge wave
                dispatch, ``skeletonize()`` additionally rescales this
                per-wave inside each worker (``dask_util.
                rescale_idi_concurrency``) to that worker's fair share of
                its *job's* real CPU affinity, since a job's cpuset is
                shared by every worker process inside it.
            sharded: Write outputs as neuroglancer_uint64_sharded_v1 instead of
                     one file per ID. Workers still write per-ID files during
                     the dask phase; the driver repacks them into shards at the
                     end and deletes the originals.
            shard_bits, minishard_bits: Sharding spec parameters; defaults give
                     2 shard files × 64 minishards. Identity hash with
                     preshift_bits=0 (good fit for densely-numbered MWS IDs).
            retry_on_oom: Halve processes-per-slot and retry on worker OOM.
            memory_retry_max: Max OOM-driven retries before raising.
            peak_bytes_baseline, peak_bytes_per_voxel: Estimator constants for
                     per-ID peak RSS. The per-voxel cost is automatically
                     bumped to ``max(dtype.itemsize + 1, peak_bytes_per_voxel)``
                     based on the segmentation's dtype, because the brief
                     read-time peak (``raw_data + bool_mask``) becomes the
                     binding constraint for uint64-stored datasets — observed
                     amplification on uint64 c-elegans data is ~10 B/voxel
                     versus ~6.6 B/voxel on uint16 data. The default of 6.63
                     B/voxel is the "honest" fit on a heavily-proofread
                     uint16 dataset; with the dtype bump it becomes ~9 B/voxel
                     on uint64 datasets automatically.
            memory_safety_multiplier: Multiplier applied to the per-ID peak
                     estimate before wave planning. Default 2.0 covers the
                     variance we've seen across c-elegans datasets (compact
                     proofread vs sparse single-pass-cleanup). Bump higher
                     (e.g. 3-4) for datasets where the giants OOM repeatedly,
                     or lower (e.g. 1.0-1.5) for tightly-profiled datasets
                     where you want maximum throughput.
            memory_fraction: Fraction of per-slot memory considered usable
                     when planning waves (rest is dask/OS/library overhead).
            write_vertex_radius: When True, write the EDT-sampled per-vertex
                     radius as a float32 ``radius`` vertex attribute on the
                     full skeletons (declared in the full ``info`` so
                     neuroglancer can color by it). Default False keeps the
                     full skeletons geometry-only. The geometrically-simplified
                     skeletons never carry it (simplify drops vertices, so the
                     radii would be lossy); the prune-only output does (see
                     ``prune_only``).
            prune_only: When True, the second output is a *prune-only*
                     skeleton (short terminal branches removed, but every
                     surviving original vertex kept -- no geometric
                     simplification), written to a ``pruned/`` directory
                     instead of ``simplified/``. ``tolerance_nm`` is ignored.
                     Because no vertices move or merge, each surviving node
                     keeps its exact original EDT radius, so this output also
                     carries the per-vertex radius when ``write_vertex_radius``
                     is set. Default False preserves the prune+simplify
                     ``simplified/`` output.
        """
        super().__init__(num_workers)
        self.concurrency_limit = dask_util.resolve_concurrency_limit(
            num_workers, concurrency_limit
        )
        self.segmentation_path = segmentation_path
        self.segmentation_idi = ImageDataInterface(
            segmentation_path,
            timeout=timeout,
            concurrency_limit=self.concurrency_limit,
        )
        self.output_path = str(output_path).rstrip("/")

        if csv_path is None:
            csv_path = self._generate_bbox_csv()
        elif not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"csv_path {csv_path!r} does not exist. Pass csv_path=None "
                f"to auto-generate bboxes via Measure, or provide an "
                f"existing CSV (the kind Measure produces)."
            )
        self.csv_path = csv_path
        # Resolve the pre-skeletonization morphological operations. The newer
        # ``morphological_operations`` (a sequence of ops) supersedes the
        # legacy ``erosion`` shorthand when provided; otherwise ``erosion`` is
        # mapped into the equivalent op list.
        if morphological_operations is None:
            ops_spec = _erosion_to_operations(erosion)
        else:
            ops_spec = morphological_operations
        self.morphological_operations = normalize_morphological_operations(ops_spec)
        self.min_branch_length_nm = min_branch_length_nm
        self.tolerance_nm = tolerance_nm
        self.num_workers = num_workers
        self.sharded = sharded
        self.shard_bits = shard_bits
        self.minishard_bits = minishard_bits
        self.retry_on_oom = retry_on_oom
        self.memory_retry_max = memory_retry_max
        self.peak_bytes_baseline = float(peak_bytes_baseline)
        self.peak_bytes_per_voxel = float(peak_bytes_per_voxel)
        self.memory_safety_multiplier = float(memory_safety_multiplier)
        self.memory_fraction = float(memory_fraction)
        self.skeleton_properties = self._normalize_skeleton_properties(
            skeleton_properties
        )
        self.write_vertex_radius = bool(write_vertex_radius)
        self.prune_only = bool(prune_only)
        # The second output is either prune+simplify ("simplified") or, when
        # prune_only is set, prune-only ("pruned"). Named honestly so the
        # neuroglancer layer reflects what it actually contains.
        self.second_subdir = "pruned" if self.prune_only else "simplified"
        # Per-instance suffix so concurrent runs sharing output_path don't
        # collide on the wave merge dirs.
        self._run_id = uuid.uuid4().hex[:8]

        # Load CSV with bounding box info
        self.bbox_df = pd.read_csv(csv_path, index_col=0)
        self.ids = self.bbox_df.index.tolist()

        # Create output directories
        # Each output directory (full and the second output) needs its own
        # structure. The second is 'simplified' or, in prune_only mode,
        # 'pruned'.
        os.makedirs(f"{output_path}/full", exist_ok=True)
        os.makedirs(f"{output_path}/full/segment_properties", exist_ok=True)
        os.makedirs(f"{output_path}/{self.second_subdir}", exist_ok=True)
        os.makedirs(
            f"{output_path}/{self.second_subdir}/segment_properties", exist_ok=True
        )

        logger.info(f"Loaded {len(self.ids)} IDs from {csv_path}")
        logger.info(f"Output will be written to {output_path}")

    def _generate_bbox_csv(self):
        """Run Measure on the segmentation to produce a per-object bbox CSV.

        Used when the caller does not supply ``csv_path``. The CSV lands at
        ``<output_path>/bboxes/<leaf>.csv`` so the user can find/reuse it
        later (point a subsequent run at it via ``csv_path``).
        """
        from cellmap_analyze.analyze.measure import Measure
        from cellmap_analyze.util.io_util import get_leaf_name_from_path

        bbox_dir = os.path.join(self.output_path, "bboxes")
        os.makedirs(bbox_dir, exist_ok=True)
        logger.info(
            "No csv_path provided; running Measure on %s to generate "
            "per-object bboxes (output -> %s).",
            self.segmentation_path,
            bbox_dir,
        )
        Measure(
            input_path=self.segmentation_path,
            output_path=bbox_dir,
            num_workers=self.num_workers,
        ).get_measurements()

        leaf = get_leaf_name_from_path(self.segmentation_path) or "measurements"
        csv_path = os.path.join(bbox_dir, f"{leaf}.csv")
        if not os.path.exists(csv_path):
            raise RuntimeError(
                f"Measure did not produce expected CSV at {csv_path}"
            )
        logger.info("Auto-generated bbox CSV at %s", csv_path)
        return csv_path

    @staticmethod
    def _empty_metrics():
        return {
            "longest_shortest_path_nm": 0.0,
            "num_branches": 0,
            "radius_mean_nm": np.nan,
            "radius_std_nm": np.nan,
        }

    @staticmethod
    def calculate_id_skeleton(
        id_value,
        segmentation_idi: ImageDataInterface,
        bbox_df: pd.DataFrame,
        output_path: str,
        morphological_operations: list,
        min_branch_length_nm: float,
        tolerance_nm: float,
        sharded: bool = False,
        write_vertex_radius: bool = False,
        prune_only: bool = False,
        second_subdir: str = "simplified",
        processes_per_job: int = None,
    ):
        """
        Process a single ID: extract, skeletonize, prune, (simplify,) and emit.

        Emits two skeletons: ``full`` (raw) and ``second_subdir`` -- either
        ``simplified`` (prune+simplify) or, when ``prune_only`` is set,
        ``pruned`` (prune only, all surviving original vertices kept).

        When ``sharded=True``, encoded skeleton bytes are returned in the
        result dict under ``"full_bytes"``/``"{second_subdir}_bytes"`` so the
        driver can pack them into shard files via the existing pickle merge
        path — no per-ID NRS write happens. When ``sharded=False``, per-ID
        files are written under ``{output_path}/{full,<second_subdir>}/{id}``.

        processes_per_job: See ``SplitNarrowBridges.split_id``'s parameter
            of the same name -- rescales ``segmentation_idi``'s
            concurrency_limit to this worker's fair share of its job's real
            CPU affinity (``dask_util.rescale_idi_concurrency``). None
            (default) leaves it untouched.
        """
        from funlib.geometry import Roi

        dask_util.rescale_idi_concurrency((segmentation_idi,), processes_per_job)

        result: dict = dict(Skeletonize._empty_metrics())

        def emit(subdir: str, skel_obj: CustomSkeleton):
            # Radii are written on the full skeleton and on the prune-only
            # output (both keep every original vertex, so radii stay exact and
            # aligned). The geometrically-simplified output stays attribute-free
            # -- simplify drops vertices, so its radii would be lossy and its
            # info declares no vertex attributes.
            include_radii = write_vertex_radius and (
                subdir == "full" or prune_only
            )
            encoded = skel_obj.encode_neuroglancer_bytes(include_radii=include_radii)
            if sharded:
                result[f"{subdir}_bytes"] = encoded
            else:
                path = f"{output_path}/{subdir}/{id_value}"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(encoded)

        def emit_empty():
            empty = CustomSkeleton(vertices=[], edges=[])
            emit("full", empty)
            emit(second_subdir, empty)

        try:
            # Get bounding box for this ID
            row = bbox_df.loc[id_value]
            min_x = row["MIN X (nm)"]
            min_y = row["MIN Y (nm)"]
            min_z = row["MIN Z (nm)"]
            max_x = row["MAX X (nm)"]
            max_y = row["MAX Y (nm)"]
            max_z = row["MAX Z (nm)"]

            # Create ROI with 1-voxel padding
            # Bounding box coords from CSV are in true nm; convert to scaled coordinates.
            # MIN/MAX (nm) are voxel-*center* coordinates (see measure_util.py), not
            # edges. `start_point` is deliberately left as `MIN*sf - padding` (not
            # shifted to the true low edge first): the vertex-coordinate math below
            # adds this same value directly to `index * spacing`-style local
            # coordinates from skimage_to_custom_skeleton_fast (an edge-relative
            # convention), and MIN*sf already sitting half a voxel inside the true
            # edge is exactly the correction needed to land on voxel *centers*
            # (see the OME-translation regression test in test_skeletonize.py).
            # Changing this breaks vertex coordinates, not just the read margin.
            #
            # On the high side there's no such coupling (`end_point` only controls
            # how much extra background the read grabs, not any offset used later),
            # so it gets the real fix: MAX*sf + padding only reaches the far edge of
            # the object's own last voxel (MAX is a center coordinate, already half
            # a voxel inside the true high edge), giving ~zero actual margin there.
            # Shifting by the missing half voxel first makes it a genuine full
            # voxel of background beyond the object.
            voxel_size = np.array(segmentation_idi.voxel_size, dtype=float)
            sf = segmentation_idi.voxel_size_scale_factor
            padding = voxel_size  # 1 voxel in each direction
            start_point = np.array([min_z * sf, min_y * sf, min_x * sf]) - padding
            end_point = (
                np.array([max_z * sf, max_y * sf, max_x * sf]) + voxel_size / 2 + padding
            )
            roi = Roi(start_point, end_point - start_point)

            logger.info(f"Processing ID {id_value}: ROI {roi}")

            # Read data for this ID
            data = segmentation_idi.to_ndarray_ts(roi)
            data = data == id_value

            # Check if there's any data
            if not np.any(data):
                logger.warning(f"No voxels found for ID {id_value}, emitting empty skeleton")
                emit_empty()
                return result

            # Resample to isotropic if needed so skeletonize thins uniformly
            # Use original (true nm) voxel_size for physical operations
            original_vs = segmentation_idi.original_voxel_size
            min_voxel = min(original_vs)
            is_anisotropic = not all(v == min_voxel for v in original_vs)
            if is_anisotropic:
                zoom_factors = tuple(v / min_voxel for v in original_vs)
                data = zoom(data, zoom_factors, order=0)
                isotropic_voxel_size = np.array([min_voxel] * 3)
            else:
                isotropic_voxel_size = np.array(original_vs)

            # Apply the requested morphological operations (in order) to the
            # mask that gets skeletonized.
            original_mask = data
            if morphological_operations:
                data = apply_morphological_operations(
                    data, morphological_operations
                )

            # Compute EDT for radii on the union of the original and processed
            # masks. For shrinking ops (erosion/opening) the processed mask is
            # a subset, so the union is the original -- radii reflect the true
            # object thickness (unchanged from the pre-erosion behavior). For
            # growing ops (dilation/closing) the union is the processed mask,
            # so radii reflect the grown/hole-filled structure that was
            # actually skeletonized.
            edt_mask = (
                np.logical_or(original_mask, data)
                if morphological_operations
                else data
            )
            distance_transform = edt_module.edt(
                edt_mask, anisotropy=tuple(isotropic_voxel_size)
            )

            if morphological_operations and not np.any(data):
                logger.warning(
                    f"Morphological operations removed all voxels for ID "
                    f"{id_value}, emitting empty skeleton"
                )
                emit_empty()
                return result

            # Skeletonize using Lee's algorithm (skimage default). It has
            # known limitations (e.g. thin structures may lose branches) but
            # is sufficient for now.
            skel = skeletonize(data)

            if not np.any(skel):
                # Lee's 3D thinning algorithm peels mirror-symmetrically and
                # can wipe out compact/spherical/cuboidal objects entirely.
                # When that happens but the (pre-erosion) EDT still has signal,
                # fall back to a single seed vertex at the EDT peak (the
                # most-interior voxel). The object then gets a meaningful
                # position and a radius equal to the local half-thickness,
                # even though longest_shortest_path stays 0 (single point).
                peak_idx = np.unravel_index(
                    int(np.argmax(distance_transform)), distance_transform.shape
                )
                peak_radius_nm = float(distance_transform[peak_idx])
                if peak_radius_nm > 0:
                    logger.warning(
                        f"Skeletonization produced no voxels for ID {id_value}, "
                        f"emitting single seed vertex at EDT peak (radius={peak_radius_nm:.1f} nm)"
                    )
                    local_zyx_nm = np.array(peak_idx) * isotropic_voxel_size
                    start_point_nm = np.array(start_point) / sf
                    seed_vertex = (
                        float(local_zyx_nm[2] + start_point_nm[2]),
                        float(local_zyx_nm[1] + start_point_nm[1]),
                        float(local_zyx_nm[0] + start_point_nm[0]),
                    )
                    seed_skel = CustomSkeleton(
                        vertices=[seed_vertex],
                        edges=np.zeros((0, 2), dtype=np.uint32),
                    )
                    # Keep the radius attribute populated even for the single
                    # seed vertex (set directly to dodge add_vertex's
                    # falsy-radius skip) so the full / prune-only outputs stay
                    # consistent with their declared vertex attribute.
                    if write_vertex_radius:
                        seed_skel.radii = [peak_radius_nm]
                    emit("full", seed_skel)
                    emit(second_subdir, seed_skel)
                    result["radius_mean_nm"] = peak_radius_nm
                    result["radius_std_nm"] = 0.0
                    return result

                logger.warning(
                    f"Skeletonization produced no voxels for ID {id_value}, emitting empty skeleton"
                )
                emit_empty()
                return result

            # Sample radii at skeleton voxel positions
            skel_coords = np.argwhere(skel)
            radii = distance_transform[
                skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]
            ]

            # Convert to custom skeleton format
            # spacing parameter scales the vertices by voxel_size
            skeleton = skimage_to_custom_skeleton_fast(
                skel, spacing=isotropic_voxel_size
            )

            # Transform vertices: add ROI offset and swap Z/X for neuroglancer (ZYX -> XYZ)
            # Vertices from skimage_to_custom_skeleton_fast are in true nm (local coords)
            # start_point is in scaled coords, convert back to true nm
            start_point_nm = np.array(start_point) / sf
            skeleton.vertices = [
                tuple(
                    [
                        v[2] + start_point_nm[2],
                        v[1] + start_point_nm[1],
                        v[0] + start_point_nm[0],
                    ]
                )
                for v in skeleton.vertices
            ]

            # Extract polylines with transformed coordinates
            # This ensures that prune() and simplify() have access to polylines
            # with the correct global coordinates
            g = skeleton.skeleton_to_graph()
            skeleton.polylines = skeleton.get_polylines_positions_from_graph(g)

            # Attach the per-vertex radii (sampled from the EDT in the same
            # np.argwhere voxel order that produced the skeleton vertices)
            # before pruning. prune() carries each surviving node's exact
            # original radius through (graph_to_skeleton rebuilds vertices and
            # radii from the same node set), so the pruned skeleton's radii are
            # available both for the prune-only output and for the radius
            # stats below. This is independent of write_vertex_radius, which
            # only controls whether radii are written into the skeleton bytes.
            skeleton.radii = list(radii)

            # Prune
            if min_branch_length_nm > 0:
                pruned = skeleton.prune(min_branch_length_nm)
            else:
                pruned = skeleton

            # Compute skeleton metrics on pruned skeleton
            num_branches = len(pruned.polylines)
            longest_shortest_path = 0.0
            if len(pruned.vertices) > 1:
                pruned_graph = pruned.skeleton_to_graph()
                for component in nx.connected_components(pruned_graph):
                    if len(component) < 2:
                        continue
                    subgraph = pruned_graph.subgraph(component)
                    start = next(iter(component))
                    lengths = nx.single_source_dijkstra_path_length(
                        subgraph, start, weight="weight"
                    )
                    far_node = max(lengths, key=lengths.get)
                    lengths2 = nx.single_source_dijkstra_path_length(
                        subgraph, far_node, weight="weight"
                    )
                    component_diameter = max(lengths2.values())
                    longest_shortest_path = max(
                        longest_shortest_path, component_diameter
                    )

            # Radius stats describe the pruned skeleton, consistent with
            # num_branches / longest_shortest_path above. Fall back to the
            # full sampled radii if pruning somehow dropped them.
            pruned_radii = pruned.radii if pruned.radii else radii
            result["longest_shortest_path_nm"] = longest_shortest_path
            result["num_branches"] = num_branches
            result["radius_mean_nm"] = float(np.mean(pruned_radii))
            result["radius_std_nm"] = float(np.std(pruned_radii))

            # Build the second output: prune-only keeps every surviving
            # vertex (and its exact radius); otherwise simplify the pruned
            # skeleton. tolerance_nm is ignored in prune_only mode.
            if prune_only:
                second_skel = pruned
            elif tolerance_nm > 0:
                second_skel = pruned.simplify(tolerance_nm)
            else:
                second_skel = pruned

            if len(second_skel.vertices) == 0:
                logger.warning(
                    f"Pruning/simplification removed all vertices for ID {id_value}, emitting empty skeleton"
                )
                emit_empty()
                return result

            # Ensure edges are properly shaped numpy arrays before encoding
            # (single vertex / empty edges case).
            if len(skeleton.edges) == 0:
                skeleton.edges = np.zeros((0, 2), dtype=np.uint32)
            else:
                skeleton.edges = np.array(skeleton.edges, dtype=np.uint32)

            if len(second_skel.edges) == 0:
                second_skel.edges = np.zeros((0, 2), dtype=np.uint32)
            else:
                second_skel.edges = np.array(second_skel.edges, dtype=np.uint32)

            emit("full", skeleton)
            emit(second_subdir, second_skel)
            return result

        except Exception as e:
            logger.error(f"Error processing ID {id_value}: {e}", exc_info=True)
            raise

    def write_neuroglancer_info_files(self):
        """
        Write the neuroglancer info file and segment_properties info file for both full and simplified directories.
        """
        # Write info files for the 'full' and second ('simplified'/'pruned')
        # output directories.
        for subdir in ["full", self.second_subdir]:
            # Write main info file for skeletons
            info = {
                "@type": "neuroglancer_skeletons",
                "transform": [
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                ],  # Identity transform since we're using physical coordinates
                "segment_properties": "segment_properties",
            }

            # The full and prune-only skeletons carry a per-vertex radius
            # (sampled from the EDT) when write_vertex_radius is set; declare
            # it so neuroglancer can read and color by it. The
            # geometrically-simplified skeletons are geometry-only.
            if self.write_vertex_radius and (subdir == "full" or self.prune_only):
                info["vertex_attributes"] = [
                    {
                        "id": "radius",
                        "data_type": "float32",
                        "num_components": 1,
                    }
                ]

            if self.sharded:
                from cellmap_analyze.util.sharded_skeleton import make_sharding_spec
                info["sharding"] = make_sharding_spec(
                    shard_bits=self.shard_bits, minishard_bits=self.minishard_bits
                )

            info_path = f"{self.output_path}/{subdir}/info"
            with open(info_path, "w") as f:
                json.dump(info, f)
            logger.info(f"Wrote neuroglancer info file to {info_path}")

            # Initial (metrics-free) segment_properties so the layer is valid
            # even before metrics are computed. We rewrite it with the numeric
            # properties at the end of skeletonize().
            self._write_segment_properties_info(subdir, metrics_by_id=None)

    # Per-ID skeleton metrics that can be baked into the neuroglancer
    # segment_properties as sortable side-panel columns. The defaults are the
    # two that triage best (how complex / how long); radius stats are opt-in.
    _SKELETON_PROPERTY_DEFS = {
        "num_branches": {
            "data_type": "int32",
            "description": "Number of skeleton branches",
            "cast": int,
            "default": 0,
        },
        "longest_shortest_path_nm": {
            "data_type": "float32",
            "description": "Longest shortest path through the skeleton (nm)",
            "cast": float,
            "default": 0.0,
        },
        "radius_mean_nm": {
            "data_type": "float32",
            "description": "Mean skeleton radius from EDT (nm)",
            "cast": float,
            "default": 0.0,
        },
        "radius_std_nm": {
            "data_type": "float32",
            "description": "Std of skeleton radius (nm)",
            "cast": float,
            "default": 0.0,
        },
    }
    DEFAULT_SKELETON_PROPERTIES = (
        "num_branches",
        "longest_shortest_path_nm",
    )
    ALL_SKELETON_PROPERTIES = tuple(_SKELETON_PROPERTY_DEFS.keys())

    @classmethod
    def _normalize_skeleton_properties(cls, value):
        """Normalize the user-facing ``skeleton_properties`` argument to a
        list of metric keys to emit.

        - ``True`` (default) -> ``DEFAULT_SKELETON_PROPERTIES`` (num_branches,
          longest_shortest_path_nm). Triage-first; not the full set.
        - ``False`` / ``None`` -> ``[]`` (label only, legacy behavior).
        - ``"all"`` -> every metric in ``ALL_SKELETON_PROPERTIES``.
        - ``list``/``tuple`` of metric keys -> exactly those, validated.
        """
        if value is True:
            return list(cls.DEFAULT_SKELETON_PROPERTIES)
        if value is False or value is None:
            return []
        if isinstance(value, str):
            if value == "all":
                return list(cls.ALL_SKELETON_PROPERTIES)
            raise ValueError(
                f"skeleton_properties string must be 'all', got {value!r}; "
                f"valid keys: {cls.ALL_SKELETON_PROPERTIES}"
            )
        if isinstance(value, (list, tuple, set)):
            keys = list(value)
            unknown = [k for k in keys if k not in cls._SKELETON_PROPERTY_DEFS]
            if unknown:
                raise ValueError(
                    f"unknown skeleton_properties: {unknown}; valid keys: "
                    f"{cls.ALL_SKELETON_PROPERTIES}"
                )
            return keys
        raise TypeError(
            f"skeleton_properties must be bool, 'all', or a list/tuple of "
            f"metric keys, got {type(value).__name__}"
        )

    def _write_segment_properties_info(self, subdir, metrics_by_id=None):
        """Write the segment_properties/info file for a subdir.

        If ``metrics_by_id`` is provided and ``self.skeleton_properties`` lists
        any metric keys, bake those per-ID skeleton metrics into the file as
        ``number`` properties so they surface as sortable columns in the
        neuroglancer side panel.
        """
        segment_ids = [str(int(i)) for i in self.ids]
        properties = [
            {
                "id": "label",
                "type": "label",
                "values": ["" for _ in segment_ids],
            }
        ]
        if self.skeleton_properties and metrics_by_id:
            for key in self.skeleton_properties:
                spec = self._SKELETON_PROPERTY_DEFS[key]
                default = spec["default"]
                cast = spec["cast"]
                values = []
                for i in self.ids:
                    v = metrics_by_id.get(int(i), {}).get(key, default)
                    # JSON has no NaN; render missing/NaN as the default so
                    # neuroglancer doesn't choke and sortable columns degrade
                    # gracefully (Lee's-wiped objects sit at one end).
                    try:
                        if v != v:  # NaN check
                            v = default
                    except TypeError:
                        v = default
                    values.append(cast(v))
                properties.append(
                    {
                        "id": key,
                        "type": "number",
                        "data_type": spec["data_type"],
                        "description": spec["description"],
                        "values": values,
                    }
                )

        info = {
            "@type": "neuroglancer_segment_properties",
            "inline": {"ids": segment_ids, "properties": properties},
        }
        path = f"{self.output_path}/{subdir}/segment_properties/info"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(info, f, allow_nan=False)
        logger.info(f"Wrote segment_properties info file to {path}")

    def _estimate_peak_bytes(self, id_value):
        """Estimate per-ID peak RSS from the cached bbox row.

        Uses the isotropic voxel count of the (padded) bbox times a
        per-voxel cost that accounts for the segmentation's dtype. The
        read-time transient ``raw_data + bool_mask`` peaks at roughly
        ``dtype.itemsize + 1`` bytes per voxel; the later EDT/skel phase
        peaks at roughly ``peak_bytes_per_voxel``. Whichever is larger
        sets the per-voxel cost (max, not sum, because they occur at
        different points in time). ``memory_safety_multiplier`` then
        absorbs cross-dataset variance from morphology/cleanup level.
        """
        row = self.bbox_df.loc[id_value]
        vs_nm = self.segmentation_idi.voxel_size
        original_vs = self.segmentation_idi.original_voxel_size
        dx = (row["MAX X (nm)"] - row["MIN X (nm)"]) + 2 * vs_nm[0]
        dy = (row["MAX Y (nm)"] - row["MIN Y (nm)"]) + 2 * vs_nm[1]
        dz = (row["MAX Z (nm)"] - row["MIN Z (nm)"]) + 2 * vs_nm[2]
        min_vs = min(original_vs)
        iso_voxels = (dx * dy * dz) / (min_vs ** 3)

        dtype_bytes = self.segmentation_idi.ds.data.dtype.itemsize
        bytes_per_voxel = max(dtype_bytes + 1, self.peak_bytes_per_voxel)
        peak = self.peak_bytes_baseline + bytes_per_voxel * iso_voxels
        return int(peak * self.memory_safety_multiplier)

    def skeletonize(self):
        """
        Main method to skeletonize all IDs in parallel.

        Plans memory-aware dask waves: groups IDs by the per-ID peak RSS
        estimator (``_estimate_peak_bytes``) into waves whose ``processes``
        per slot is tuned so one item fits per worker. Each wave runs as
        its own dask cluster with the OOM-retry safety net underneath.
        """
        logger.info(f"Starting skeletonization of {len(self.ids)} IDs")

        # First write the info files (only once)
        self.write_neuroglancer_info_files()

        try:
            base_config = dask_util._load_dask_config() if self.num_workers > 1 else None
        except (FileNotFoundError, KeyError, TypeError, ValueError) as e:
            logger.warning(
                "Could not load dask-config.yaml for wave planning (%s); "
                "running all IDs in a single wave.",
                e,
            )
            base_config = None

        items = [(int(i), self._estimate_peak_bytes(i)) for i in self.ids]
        waves = dask_util.plan_memory_waves(
            items,
            self.num_workers,
            config=base_config,
            memory_fraction=self.memory_fraction,
        )
        self._log_wave_plan(waves)

        processes_per_job_by_wave = dask_util.wave_uses_shared_job_cpuset(base_config)

        tmp_merge_root = (
            f"{self.output_path}/_tmp_skeleton_metrics_to_merge_{self._run_id}"
        )
        all_metrics = []

        for wave_index, wave in enumerate(waves, start=1):
            # The outer phase_name (used by the OOM-retry log line) keeps
            # the wave's identity stable across retries. The inner msg
            # built inside _phase reflects the *current* procs/slot so the
            # "Started skeletonize..." log line is accurate after a halving.
            wave_label = f"skeletonize wave {wave_index}/{len(waves)} ({len(wave.item_ids)} IDs)"
            wave_ids = wave.item_ids
            wave_merge_dir = f"{tmp_merge_root}_wave{wave_index}"
            processes_per_job = wave.processes if processes_per_job_by_wave else None

            def _wrapper(idx, _wave_ids=wave_ids, _procs=processes_per_job):
                return self._skeletonize_id_by_value(_wave_ids[idx], _procs)

            def _phase(workers, config, _wrapper=_wrapper, _ids=wave_ids,
                       _merge=wave_merge_dir, _label=wave_label):
                current_procs = (
                    config["jobqueue"][next(iter(config["jobqueue"]))]["processes"]
                    if config and config.get("jobqueue") else 1
                )
                msg = f"{_label}, procs/slot={current_procs}"
                return dask_util.compute_blockwise_partitions(
                    len(_ids), workers, self.compute_args, logger, msg,
                    _wrapper,
                    merge_info=(Skeletonize._merge_skeleton_metrics, _merge),
                    config=config,
                )

            wave_metrics = dask_util.run_with_oom_retry(
                _phase, wave.workers, wave_label, logger,
                max_retries=self.memory_retry_max,
                retry_on_oom=self.retry_on_oom,
                config=wave.config,
            )
            all_metrics.extend(wave_metrics)

        # When sharded, workers piggybacked encoded skeleton bytes onto each
        # metrics dict via the pickle merge — no per-ID files were written.
        # Pop the bytes out (so they don't end up in the CSV) and write shards.
        if self.sharded:
            self._pack_shards_from_metrics(all_metrics)

        self._write_skeleton_csv(all_metrics)

        # Rewrite segment_properties with the per-ID metrics so they show up
        # as sortable columns in the neuroglancer side panel.
        if self.skeleton_properties:
            metrics_by_id = {int(m["id"]): m for m in all_metrics}
            for subdir in ("full", self.second_subdir):
                self._write_segment_properties_info(subdir, metrics_by_id)

        logger.info("Skeletonization complete")

    def _log_wave_plan(self, waves):
        if not waves:
            return
        total_ids = sum(len(w.item_ids) for w in waves)
        biggest = max(w.max_estimated_peak_bytes for w in waves)
        logger.info(
            "Wave plan: %d wave(s) over %d IDs (largest projected peak %.2f GB)",
            len(waves), total_ids, biggest / 1e9,
        )
        for i, wave in enumerate(waves, start=1):
            logger.info(
                "  wave %d/%d: processes/slot=%d, workers=%d, IDs=%d, "
                "max projected peak %.2f GB",
                i, len(waves), wave.processes, wave.workers,
                len(wave.item_ids), wave.max_estimated_peak_bytes / 1e9,
            )

    def _skeletonize_id_by_value(self, id_value, processes_per_job=None):
        """Dispatch one ID to ``calculate_id_skeleton``. Each wave dispatches
        a subset of IDs, so the wrapper takes the ID value directly rather
        than an index into ``self.ids``."""
        result = Skeletonize.calculate_id_skeleton(
            id_value,
            self.segmentation_idi,
            self.bbox_df,
            self.output_path,
            self.morphological_operations,
            self.min_branch_length_nm,
            self.tolerance_nm,
            sharded=self.sharded,
            write_vertex_radius=self.write_vertex_radius,
            prune_only=self.prune_only,
            second_subdir=self.second_subdir,
            processes_per_job=processes_per_job,
        )
        if result is None:
            result = Skeletonize._empty_metrics()
        result["id"] = id_value
        return result

    def _pack_shards_from_metrics(self, metrics_list):
        """Pack encoded skeleton bytes (already in memory via pickle merge)
        into precomputed sharded shard files.

        Pops ``full_bytes``/``{second_subdir}_bytes`` from each metric dict in
        ``metrics_list`` so subsequent CSV writing sees only metric columns.
        No NRS read/unlink work — the bytes were carried back on the dask
        merge path that runs for every job regardless.
        """
        import time
        from cellmap_analyze.util.sharded_skeleton import pack_sharded_skeletons

        for subdir, bytes_key in [
            ("full", "full_bytes"),
            (self.second_subdir, f"{self.second_subdir}_bytes"),
        ]:
            dir_path = f"{self.output_path}/{subdir}"

            t0 = time.time()
            id_to_bytes: dict[int, bytes] = {}
            iterator = tqdm(
                metrics_list,
                desc=f"Gathering {subdir} skeleton bytes",
                unit="id",
            )
            for m in iterator:
                data = m.pop(bytes_key, None)
                if data is None:
                    continue
                id_to_bytes[int(m["id"])] = data
            logger.info(
                f"Gathered {len(id_to_bytes)} {subdir} skeletons in "
                f"{time.time() - t0:.1f}s"
            )

            if not id_to_bytes:
                logger.warning(
                    f"No {subdir} skeleton bytes found in metrics; skipping shard pack"
                )
                continue

            t0 = time.time()
            pack_sharded_skeletons(
                id_to_bytes,
                dir_path,
                shard_bits=self.shard_bits,
                minishard_bits=self.minishard_bits,
            )
            logger.info(
                f"Packed {len(id_to_bytes)} {subdir} skeletons into shards under "
                f"{dir_path} in {time.time() - t0:.1f}s"
            )
            del id_to_bytes

    @staticmethod
    def _merge_skeleton_metrics(list_of_results):
        merged = []
        for result in list_of_results:
            merged.append(result)
        return merged

    def _write_skeleton_csv(self, skeleton_metrics):
        original_df = pd.read_csv(self.csv_path, index_col=0)
        metrics_df = pd.DataFrame(skeleton_metrics)
        metrics_df = metrics_df.set_index("id")
        metrics_df = metrics_df.rename(
            columns={
                "longest_shortest_path_nm": "Longest Shortest Path (nm)",
                "num_branches": "Number of Branches",
                "radius_mean_nm": "Radius Mean (nm)",
                "radius_std_nm": "Radius Std (nm)",
            }
        )
        combined_df = original_df.join(metrics_df)
        csv_dir = os.path.dirname(self.csv_path)
        csv_basename = os.path.splitext(os.path.basename(self.csv_path))[0]
        output_csv = os.path.join(csv_dir, f"{csv_basename}_with_skeletons.csv")
        combined_df.to_csv(output_csv)
        logger.info(f"Wrote skeleton metrics CSV to {output_csv}")


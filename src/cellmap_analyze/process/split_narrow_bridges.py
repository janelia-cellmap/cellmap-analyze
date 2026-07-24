import logging
import os
import shutil
import uuid
from abc import ABC, abstractmethod

import edt as edt_module
import fastremap
import networkx as nx
import numpy as np
import pandas as pd
from funlib.geometry import Roi
from scipy.ndimage import binary_dilation
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

from cellmap_analyze.util import dask_util, io_util
from cellmap_analyze.util.dask_util import create_block_from_index
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.io_util import get_output_path_from_input_path
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.skeleton_util import skimage_to_custom_skeleton_fast
from cellmap_analyze.util.zarr_util import create_multiscale_dataset_idi

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- Split strategies --------------------------------------------------
#
# A strategy only decides *where* to cut a single object's binary mask; all
# I/O, global ID bookkeeping, and blockwise writing lives in
# SplitNarrowBridges. See docs/split_narrow_bridges_plan.md for the design
# rationale (why this is pluggable, and the open questions around
# thresholding).


class SplitStrategy(ABC):
    @abstractmethod
    def find_subpieces(
        self, mask, voxel_size, neck_radius_voxels, minimum_subregion_volume_voxels
    ):
        """Partition a single object's binary mask into candidate subpieces.

        Args:
            mask: 3D boolean array, the object's voxels within its padded bbox.
            voxel_size: physical (original, possibly anisotropic) voxel size.
            neck_radius_voxels: candidate-cut threshold, in voxels (isotropic
                approximation -- see EDTWatershedSplit).
            minimum_subregion_volume_voxels: optional accept-gate; None
                disables it.

        Returns:
            Integer label array, same shape as ``mask``: 0 outside the mask,
            1..k inside marking k candidate subpieces. k <= 1 means "no
            split" (the caller treats this as a no-op).
        """
        raise NotImplementedError


def _merge_small_fragments(labels, min_volume_voxels):
    """Iteratively dissolve the smallest fragment below ``min_volume_voxels``
    into its largest-bordering neighbor, until every surviving fragment
    clears the gate (or only one fragment remains).

    Never drops a voxel -- a fragment with no foreground neighbor (shouldn't
    happen for a single connected object, but morphology is messy) merges
    into the overall largest fragment instead.
    """
    labels = labels.copy()
    while True:
        ids, counts = fastremap.unique(labels[labels > 0], return_counts=True)
        if len(ids) <= 1:
            break
        min_idx = np.argmin(counts)
        if counts[min_idx] >= min_volume_voxels:
            break
        smallest_id = ids[min_idx]
        dilated = binary_dilation(labels == smallest_id)
        neighbor_mask = dilated & (labels != smallest_id) & (labels > 0)
        neighbor_values = labels[neighbor_mask]
        if neighbor_values.size == 0:
            neighbor_id = ids[np.argmax(counts)]
        else:
            neighbor_ids, neighbor_counts = fastremap.unique(
                neighbor_values, return_counts=True
            )
            neighbor_id = neighbor_ids[np.argmax(neighbor_counts)]
        labels[labels == smallest_id] = neighbor_id
    return labels


def _merge_thick_boundaries(labels, distance, neck_radius_nm):
    """Reject any watershed split whose boundary isn't actually thin.

    Two peaks sitting in one continuously-fat, irregularly-shaped blob (a
    lumpy real nucleus, not two merged objects) can each seed their own
    watershed basin even with no real neck between them -- a size gate alone
    doesn't catch this. This finds, for every pair of adjacent regions, the
    widest point along their shared boundary (the "pass" between the two
    basins); if it isn't meaningfully thinner than ``neck_radius_nm``, the
    regions get merged back together (worst -- thickest -- offender first)
    until every remaining boundary is thin enough.

    A first version of this recomputed every region's mask + dilation +
    neighbor boundaries from scratch after every single merge -- fine for a
    handful of small fragments, but on a real ~100M-voxel merged-nucleus
    object with ~50 initial peaks it took 2.4 *hours*, because most of that
    work (whole-array boolean ops and dilations) was redone from a cold
    start for every one of the ~20 merges needed to converge. Fixed by
    separating the (expensive, but one-time) voxel-array work from the
    (cheap, iterative) decision-making: find every touching pair of labels
    and the max EDT radius at their shared boundary in a single pass over
    the array, then do all the merging on that small label-adjacency graph
    (tens of entries, not tens of millions of voxels) via union-find,
    folding a merged region's neighbor list into its surviving root instead
    of rescanning the array.
    """
    ids = fastremap.unique(labels[labels > 0])
    if len(ids) <= 1:
        return labels

    # Single pass over the array: for every face-adjacent pair of touching,
    # differing labels, record the largest EDT radius seen at that junction
    # (the widest point along their shared boundary).
    pass_radius = {}
    for axis in range(labels.ndim):
        lo_slice = [slice(None)] * labels.ndim
        hi_slice = [slice(None)] * labels.ndim
        lo_slice[axis] = slice(0, -1)
        hi_slice[axis] = slice(1, None)
        lo_slice, hi_slice = tuple(lo_slice), tuple(hi_slice)
        a, b = labels[lo_slice], labels[hi_slice]
        touching = (a > 0) & (b > 0) & (a != b)
        if not np.any(touching):
            continue
        la, lb = a[touching], b[touching]
        radii = np.maximum(distance[lo_slice][touching], distance[hi_slice][touching])
        lo_id, hi_id = np.minimum(la, lb), np.maximum(la, lb)
        for l, h, r in zip(lo_id.tolist(), hi_id.tolist(), radii.tolist()):
            key = (l, h)
            if r > pass_radius.get(key, -1.0):
                pass_radius[key] = r

    if not pass_radius:
        return labels

    # adjacency[a][b] = widest boundary radius between current regions a, b.
    # Keys are always *current* union-find roots -- merging folds b's entries
    # into a's and fixes up any of b's former neighbors to point at a.
    adjacency = {}
    for (lo_id, hi_id), radius in pass_radius.items():
        adjacency.setdefault(lo_id, {})[hi_id] = radius
        adjacency.setdefault(hi_id, {})[lo_id] = radius

    parent = {int(i): int(i) for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    while True:
        worst_pair, worst_radius = None, -1.0
        for a, neighbors in adjacency.items():
            for b, radius in neighbors.items():
                if radius > worst_radius:
                    worst_radius, worst_pair = radius, (a, b)
        if worst_pair is None or worst_radius < neck_radius_nm:
            break
        a, b = worst_pair
        parent[b] = a
        b_neighbors = adjacency.pop(b)
        a_neighbors = adjacency[a]
        del a_neighbors[b]
        for other, radius in b_neighbors.items():
            if other == a:
                continue
            adjacency[other].pop(b, None)
            merged_radius = max(radius, a_neighbors.get(other, -1.0))
            a_neighbors[other] = merged_radius
            adjacency[other][a] = merged_radius

    remap = {i: find(i) for i in parent if find(i) != i}
    if not remap:
        return labels
    labels = labels.copy()
    fastremap.remap(labels, remap, preserve_missing_labels=True, in_place=True)
    return labels


class EDTWatershedSplit(SplitStrategy):
    """Distance-transform watershed: standard approach for splitting
    blob-like objects (nuclei, cells) joined by a thin neck.

    ``neck_radius_voxels`` does double duty: it sets the minimum separation
    between watershed seeds (an isotropic approximation using the finest
    voxel axis -- see the open question in docs/split_narrow_bridges_plan.md
    about anisotropic handling), *and* it's the maximum boundary width for a
    watershed split to be accepted as a genuine neck (see
    ``_merge_thick_boundaries``) -- this is the "thin, between two thicker
    things" check. ``minimum_subregion_volume_voxels`` is an optional
    second, independent gate: fragments below it are dissolved into their
    largest neighbor regardless of boundary thinness (catches tiny noise
    fragments that happen to sit behind a technically-thin boundary).
    """

    def find_subpieces(
        self, mask, voxel_size, neck_radius_voxels, minimum_subregion_volume_voxels
    ):
        if not np.any(mask):
            return np.zeros(mask.shape, dtype=np.uint8)

        distance = edt_module.edt(mask, anisotropy=tuple(voxel_size))
        min_distance = max(1, int(round(neck_radius_voxels)))
        coords = peak_local_max(
            distance,
            min_distance=min_distance,
            labels=mask.astype(np.int32),
            exclude_border=False,
        )
        if len(coords) == 0:
            return mask.astype(np.uint8)

        markers = np.zeros(mask.shape, dtype=np.int32)
        for i, coord in enumerate(coords):
            markers[tuple(coord)] = i + 1

        labels = watershed(-distance, markers, mask=mask)

        # neck_radius_voxels is an isotropic voxel-count approximation
        # (neck_radius_nm / min(voxel_size)); round-trip it back to physical
        # nm to compare against the EDT-based (physical) boundary radii.
        neck_radius_nm = neck_radius_voxels * min(voxel_size)
        labels = _merge_thick_boundaries(labels, distance, neck_radius_nm)

        if minimum_subregion_volume_voxels:
            labels = _merge_small_fragments(labels, minimum_subregion_volume_voxels)

        labels, _ = fastremap.renumber(labels, in_place=True)
        return labels


class SkeletonGraphSplit(SplitStrategy):
    """Skeleton/graph-based splitting.

    Not specific to any organelle type -- it's a general strategy, just one
    better suited than EDTWatershedSplit when an object is thin and branched
    (e.g. mitochondria networks), where a bare local-radius threshold can't
    distinguish a normal tubule from a merge artifact. It works fine on
    simple blobs too (a single sphere skeletonizes to ~1 point and finds no
    candidate cuts, same as EDTWatershedSplit would report "no split").

    Builds the object's skeleton, samples the EDT radius at each skeleton
    node, and looks for candidate necks: maximal connected runs of nodes
    whose radius is below ``neck_radius_voxels`` ("thin segments") that are
    flanked by thicker material on *every* side -- not just any node below
    the threshold. A bare threshold would also flag an ordinary monotonic
    taper (e.g. a blob with a thin spur has radius shrinking steadily to the
    spur's tip -- thin all the way, but never bounded by thick material on
    the far side, since there's no second object out there). A thin segment
    that runs all the way to a skeleton endpoint is exactly that case and is
    excluded outright; a thin segment must be "thin, between two thicker
    things" on every side to qualify. Candidate segments are cut
    thinnest-first: a cut is only kept if it actually disconnects the graph,
    and (when ``minimum_subregion_volume_voxels`` is set) if every resulting
    side clears the size gate -- this is the topology-aware check that lets
    a real branch point survive while a genuine bridge gets cut. Every mask
    voxel is then assigned to its nearest skeleton node via a geodesic
    (mask-respecting) watershed to produce the final per-voxel subpieces
    array.
    """

    def find_subpieces(
        self, mask, voxel_size, neck_radius_voxels, minimum_subregion_volume_voxels
    ):
        if not np.any(mask):
            return np.zeros(mask.shape, dtype=np.uint8)

        distance = edt_module.edt(mask, anisotropy=tuple(voxel_size))
        skel = skeletonize(mask)
        if not np.any(skel):
            return mask.astype(np.uint8)

        skel_coords = np.argwhere(skel)
        radii = distance[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]]

        skeleton = skimage_to_custom_skeleton_fast(skel, spacing=voxel_size)
        # Must be set before skeleton_to_graph() -- that's when node "radius"
        # attributes get populated (see CustomSkeleton.skeleton_to_graph).
        skeleton.radii = list(radii)
        g = skeleton.skeleton_to_graph()

        if g.number_of_nodes() <= 1:
            return mask.astype(np.uint8)

        # neck_radius_voxels is an isotropic voxel-count approximation
        # (neck_radius_nm / min(voxel_size), computed once in the driver);
        # round-trip it back to physical nm to compare against the EDT-based
        # (physical) node radii.
        neck_radius_nm = neck_radius_voxels * min(voxel_size)

        # Assign every mask voxel to its nearest skeleton node, respecting
        # the mask's actual shape/connectivity (a marker-based watershed on a
        # flat field is a geodesic, through-the-mask nearest-seed assignment
        # -- unlike a straight-line Euclidean nearest neighbor, it can't leak
        # across a bend or a concavity onto the "wrong" side of a shape).
        # Node identities/positions don't change as candidate edges are tried
        # below, only which nodes end up grouped together, so this is done
        # once up front: it gives an exact per-node voxel count, and summing
        # counts over a candidate component is real voxel accounting, not a
        # geometric proxy (a pi*r^2*length tube estimate was tried first and
        # overestimated a 3-voxel stub as ~6 voxels, enough to slip past the
        # size gate; a straight-line nearest neighbor was tried second and
        # leaked most of a compact cube's volume onto a thin stub's nodes).
        num_nodes = g.number_of_nodes()
        markers = np.zeros(mask.shape, dtype=np.int32)
        markers[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]] = np.arange(
            1, num_nodes + 1
        )
        node_assignment = watershed(np.zeros(mask.shape), markers, mask=mask)
        mask_coords = np.argwhere(mask)
        nearest_node_idx = (
            node_assignment[mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2]] - 1
        )
        node_voxel_counts = np.bincount(nearest_node_idx, minlength=num_nodes)

        def component_voxel_count(component_nodes):
            return int(node_voxel_counts[list(component_nodes)].sum())

        # A candidate cut is a maximal connected run of nodes below the
        # threshold ("thin segment"), not a single node -- and only a
        # segment flanked by thicker material on *every* side qualifies.
        # Nodes below threshold alone (without the flanking check) was tried
        # first: it also flags ordinary monotonic tapers -- e.g. a solid
        # cube with a thin spur has EDT radius shrinking steadily from the
        # cube's core down to the spur's tip, so the whole taper reads as
        # "thin," even though nothing bounds it on the far side (no second
        # object out there -- it's one shape, ending in free space). A
        # per-node "local minimum vs. immediate neighbors only" check was
        # tried second: too fragile near branch points and noisy single-voxel
        # radius jitter, and it missed a real, visually obvious neck on real
        # cerebellum nucleus data. A thin segment that touches a skeleton
        # endpoint (degree < 2) is exactly the monotonic-taper case and is
        # excluded outright; every other thin segment is, by construction,
        # bounded by >= threshold-radius material on every side it connects
        # to (any lower-radius neighbor would already be part of the same
        # segment) -- this is the "thin, between two thicker things" check.
        original_neighbors = {n: list(g.neighbors(n)) for n in g.nodes}

        thin_nodes = {
            n for n in g.nodes if g.nodes[n]["radius"] < neck_radius_nm
        }
        thin_components = list(nx.connected_components(g.subgraph(thin_nodes)))
        candidate_segments = sorted(
            (
                segment
                for segment in thin_components
                if not any(len(original_neighbors[n]) < 2 for n in segment)
            ),
            key=lambda segment: min(g.nodes[n]["radius"] for n in segment),
        )

        for segment in candidate_segments:
            node_attrs = {n: dict(g.nodes[n]) for n in segment}
            saved_edges = list(g.edges(segment, data=True))
            g.remove_nodes_from(segment)
            components = list(nx.connected_components(g))
            if len(components) < 2:
                # Didn't actually separate anything (e.g. part of a cycle).
                for n, attrs in node_attrs.items():
                    g.add_node(n, **attrs)
                g.add_edges_from(saved_edges)
                continue

            if minimum_subregion_volume_voxels and not all(
                component_voxel_count(c) >= minimum_subregion_volume_voxels
                for c in components
            ):
                for n, attrs in node_attrs.items():
                    g.add_node(n, **attrs)
                g.add_edges_from(saved_edges)
                continue
            # else: accepted (segment stays removed), or no size gate --
            # accept every candidate cut that actually separates the graph.
            # A disabled gate can over-split heavily-branched objects; set
            # minimum_subregion_volume_nm_3 to guard against that (see
            # docs/split_narrow_bridges_plan.md).

        final_components = list(nx.connected_components(g))
        num_pieces = len(final_components)
        if num_pieces <= 1:
            return mask.astype(np.uint8)

        node_to_piece = np.zeros(num_nodes, dtype=np.int32)
        for piece_index, component in enumerate(final_components, start=1):
            node_to_piece[list(component)] = piece_index

        # Nodes that were cut out (genuine necks) have no piece of their own
        # -- resolve each to whichever side of the cut its nearest surviving
        # original neighbor ended up on. It's a handful of voxels right at
        # the pinch point either way, so which side they land on doesn't
        # materially matter.
        for n in range(num_nodes):
            if node_to_piece[n] != 0:
                continue
            visited = {n}
            queue = list(original_neighbors[n])
            while queue:
                cur = queue.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                if node_to_piece[cur] != 0:
                    node_to_piece[n] = node_to_piece[cur]
                    break
                queue.extend(original_neighbors[cur])

        labels = np.zeros(mask.shape, dtype=np.int32)
        labels[mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2]] = (
            node_to_piece[nearest_node_idx]
        )
        return labels


_STRATEGY_REGISTRY = {
    "edt_watershed": EDTWatershedSplit,
    "skeleton_graph": SkeletonGraphSplit,
}


class SplitNarrowBridges(ComputeConfigMixin):
    def __init__(
        self,
        segmentation_path,
        output_path,
        strategy="edt_watershed",
        neck_radius_nm=0,
        minimum_subregion_volume_nm_3=None,
        max_pieces_per_object=64,
        csv_path=None,
        num_workers=10,
        timeout=5,
        chunk_shape=None,
        delete_tmp=True,
    ):
        """
        Split accidentally-merged objects at narrow bridges, preserving all
        original foreground voxels and only changing instance labels.

        Args:
            segmentation_path: Path to the input segmentation zarr dataset.
            output_path: Path to the output segmentation dataset.
            strategy: "edt_watershed" (distance-transform watershed -- the
                simpler, standard choice for blob-like objects: nuclei,
                cells) or "skeleton_graph" (topology-aware; general-purpose,
                but the one to reach for when objects are thin/branched --
                e.g. mitochondria -- where a bare local-radius threshold
                can't tell a normal tubule from a merge artifact).
            neck_radius_nm: Candidate-cut threshold, in nm. Converted to
                voxels using the finest voxel axis (isotropic approximation).
            minimum_subregion_volume_nm_3: Optional accept-gate -- a
                candidate split is only kept if the resulting subregions
                (after merging any that fall below this volume into their
                largest neighbor) still number >= 2. None (default) disables
                the gate entirely.
            max_pieces_per_object: Safety cap on how many subpieces a single
                object may split into; exceeding it raises rather than
                silently truncating.
            csv_path: Optional path to a CSV with per-object bounding boxes
                (the kind Measure produces). If None, Measure is run on the
                segmentation to generate one at
                ``<output_path>/bboxes/<leaf>.csv``.
            num_workers: Number of parallel dask workers.
            timeout: Timeout for ImageDataInterface reads.
            delete_tmp: Delete the per-object scratch directory when done.
        """
        super().__init__(num_workers)
        self.segmentation_path = segmentation_path
        self.segmentation_idi = ImageDataInterface(
            segmentation_path, timeout=timeout, chunk_shape=chunk_shape
        )
        self.output_path = str(output_path).rstrip("/")
        self.roi = self.segmentation_idi.roi

        if strategy not in _STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy {strategy!r}; valid: {list(_STRATEGY_REGISTRY)}"
            )
        self.strategy_name = strategy
        self.strategy = _STRATEGY_REGISTRY[strategy]()

        if csv_path is None:
            csv_path = self._generate_bbox_csv()
        elif not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"csv_path {csv_path!r} does not exist. Pass csv_path=None "
                f"to auto-generate bboxes via Measure, or provide an "
                f"existing CSV (the kind Measure produces)."
            )
        self.csv_path = csv_path
        self.bbox_df = pd.read_csv(csv_path, index_col=0)
        self.ids = self.bbox_df.index.tolist()

        original_voxel_size = self.segmentation_idi.original_voxel_size
        voxel_volume = float(np.prod(original_voxel_size))
        self.neck_radius_voxels = float(neck_radius_nm) / min(original_voxel_size)
        if minimum_subregion_volume_nm_3 is None:
            self.minimum_subregion_volume_voxels = None
        else:
            self.minimum_subregion_volume_voxels = float(
                minimum_subregion_volume_nm_3
            ) / voxel_volume

        self.max_pieces_per_object = int(max_pieces_per_object)
        self.delete_tmp = delete_tmp
        # Per-instance suffix so concurrent runs sharing output_path don't
        # collide on the scratch/merge dirs.
        self._run_id = uuid.uuid4().hex[:8]

        logger.info(f"Loaded {len(self.ids)} IDs from {csv_path}")
        logger.info(f"Output will be written to {self.output_path}")

    def _generate_bbox_csv(self):
        """Run Measure on the segmentation to produce a per-object bbox CSV,
        the same way Skeletonize does when csv_path is not supplied."""
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
            raise RuntimeError(f"Measure did not produce expected CSV at {csv_path}")
        logger.info("Auto-generated bbox CSV at %s", csv_path)
        return csv_path

    @staticmethod
    def _object_roi(id_value, bbox_df, segmentation_idi):
        row = bbox_df.loc[id_value]
        voxel_size = segmentation_idi.voxel_size
        sf = segmentation_idi.voxel_size_scale_factor
        padding = voxel_size  # 1 voxel in each direction
        start_point = (
            np.array(
                [row["MIN Z (nm)"] * sf, row["MIN Y (nm)"] * sf, row["MIN X (nm)"] * sf]
            )
            - padding
        )
        end_point = (
            np.array(
                [row["MAX Z (nm)"] * sf, row["MAX Y (nm)"] * sf, row["MAX X (nm)"] * sf]
            )
            + padding
        )
        return Roi(start_point, end_point - start_point)

    @staticmethod
    def split_id(
        id_value,
        segmentation_idi: ImageDataInterface,
        bbox_df: pd.DataFrame,
        strategy: SplitStrategy,
        neck_radius_voxels: float,
        minimum_subregion_volume_voxels,
        max_pieces_per_object: int,
        scratch_dir: str,
    ):
        """Process a single object: extract its mask, find candidate
        subpieces, and (if split) persist the local labeling to a small
        per-object scratch file.

        Returns a dict describing the outcome; ``split_objects`` merges
        these on the driver before doing any blockwise writing.
        """
        roi = SplitNarrowBridges._object_roi(id_value, bbox_df, segmentation_idi)
        data = segmentation_idi.to_ndarray_ts(roi)
        mask = data == id_value
        if not np.any(mask):
            logger.warning(f"No voxels found for ID {id_value}, skipping")
            return {"id": int(id_value), "split": False}

        subpieces = strategy.find_subpieces(
            mask,
            segmentation_idi.original_voxel_size,
            neck_radius_voxels,
            minimum_subregion_volume_voxels,
        )
        num_pieces = int(subpieces.max()) if subpieces.size else 0
        if num_pieces <= 1:
            return {"id": int(id_value), "split": False}
        if num_pieces > max_pieces_per_object:
            raise ValueError(
                f"Object {id_value} split into {num_pieces} pieces, exceeding "
                f"max_pieces_per_object={max_pieces_per_object}. Increase the "
                f"limit or investigate why this object fragmented so heavily."
            )

        os.makedirs(scratch_dir, exist_ok=True)
        scratch_path = f"{scratch_dir}/{id_value}.npz"
        np.savez(
            scratch_path, subpieces=subpieces.astype(np.min_scalar_type(num_pieces))
        )

        return {
            "id": int(id_value),
            "split": True,
            "num_pieces": num_pieces,
            "scratch_path": scratch_path,
            "roi_begin": tuple(roi.get_begin()),
            "roi_shape": tuple(roi.shape),
        }

    @staticmethod
    def _merge_split_results(list_of_results):
        return list(list_of_results)

    @staticmethod
    def relabel_block_with_splits(
        block_index,
        segmentation_idi: ImageDataInterface,
        output_idi: ImageDataInterface,
        split_lookup: dict,
        dtype,
    ):
        block = create_block_from_index(output_idi, block_index)
        data = segmentation_idi.to_ndarray_ts(block.write_roi).astype(dtype)

        voxel_size = output_idi.voxel_size
        for id_value, (new_id_base, scratch_path, obj_roi) in split_lookup.items():
            overlap = block.write_roi.intersect(obj_roi)
            if overlap.empty:
                continue

            with np.load(scratch_path) as npz:
                subpieces = npz["subpieces"]

            data_offset = (overlap.begin - block.write_roi.begin) / voxel_size
            obj_offset = (overlap.begin - obj_roi.begin) / voxel_size
            shape = overlap.shape / voxel_size
            data_slice = tuple(
                slice(int(o), int(o + s)) for o, s in zip(data_offset, shape)
            )
            obj_slice = tuple(
                slice(int(o), int(o + s)) for o, s in zip(obj_offset, shape)
            )

            region = data[data_slice]
            sub_region = subpieces[obj_slice]
            piece_mask = (region == id_value) & (sub_region > 0)
            if np.any(piece_mask):
                region[piece_mask] = new_id_base + sub_region[piece_mask].astype(dtype)
                data[data_slice] = region

        output_idi.ds[block.write_roi] = data

    def _write_output(self, split_lookup, dtype=None):
        if dtype is None:
            dtype = self.segmentation_idi.dtype

        output_idi = create_multiscale_dataset_idi(
            self.output_path,
            dtype=dtype,
            voxel_size=self.segmentation_idi.voxel_size,
            total_roi=self.roi,
            write_size=self.segmentation_idi.chunk_shape * self.segmentation_idi.voxel_size,
            original_voxel_size=self.segmentation_idi.original_voxel_size,
        )

        num_blocks = dask_util.get_num_blocks(self.segmentation_idi, roi=self.roi)
        dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            f"writing split output to {self.output_path}",
            SplitNarrowBridges.relabel_block_with_splits,
            self.segmentation_idi,
            output_idi,
            split_lookup,
            dtype,
        )

    def split_objects(self):
        """Main entry point: find candidate splits for every object, then
        write the output (a verbatim copy of the input except for split
        objects, whose voxels get new instance labels)."""
        tmp_merge_root = get_output_path_from_input_path(
            self.output_path, f"_tmp_split_metrics_to_merge_{self._run_id}"
        )
        scratch_dir = get_output_path_from_input_path(
            self.output_path, f"_split_scratch_{self._run_id}"
        )

        def _wrapper(idx):
            id_value = self.ids[idx]
            return SplitNarrowBridges.split_id(
                id_value,
                self.segmentation_idi,
                self.bbox_df,
                self.strategy,
                self.neck_radius_voxels,
                self.minimum_subregion_volume_voxels,
                self.max_pieces_per_object,
                scratch_dir,
            )

        with io_util.TimingMessager("Finding candidate splits", logger):
            results = dask_util.compute_blockwise_partitions(
                len(self.ids),
                self.num_workers,
                self.compute_args,
                logger,
                f"splitting narrow bridges for {self.segmentation_path}",
                _wrapper,
                merge_info=(SplitNarrowBridges._merge_split_results, tmp_merge_root),
            )

        split_results = [r for r in results if r["split"]]
        logger.info(f"{len(split_results)}/{len(self.ids)} objects split")

        try:
            if not split_results:
                self._write_output({})
                return

            split_results.sort(key=lambda r: r["id"])
            max_original_id = max(self.ids) if self.ids else 0
            split_lookup = {}
            for rank, r in enumerate(split_results):
                new_id_base = max_original_id + 1 + rank * self.max_pieces_per_object
                obj_roi = Roi(r["roi_begin"], r["roi_shape"])
                split_lookup[r["id"]] = (new_id_base, r["scratch_path"], obj_roi)

            max_new_id = max_original_id + len(split_results) * self.max_pieces_per_object
            new_dtype = np.min_scalar_type(max_new_id)

            self._write_output(split_lookup, new_dtype)
        finally:
            if self.delete_tmp:
                shutil.rmtree(scratch_dir, ignore_errors=True)

        logger.info("Splitting complete")

import logging
import os
import shutil
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict

import edt as edt_module
import fastremap
import networkx as nx
import numpy as np
import pandas as pd
from funlib.geometry import Roi
from scipy.ndimage import binary_dilation
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

from cellmap_analyze.process.clean_connected_components import CleanConnectedComponents
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
        self,
        mask,
        voxel_size,
        neck_radius_voxels,
        minimum_subregion_volume_voxels,
        distance=None,
        neck_radius_mode="fixed",
    ):
        """Partition a single object's binary mask into candidate subpieces.

        Args:
            mask: 3D boolean array, the object's voxels within its padded bbox.
            voxel_size: physical (original, possibly anisotropic) voxel size.
            neck_radius_voxels: candidate-cut threshold, in voxels (isotropic
                approximation -- see EDTWatershedSplit). When
                ``neck_radius_mode="adaptive"``, this instead acts as a
                floor on the per-object threshold (see below).
            minimum_subregion_volume_voxels: optional accept-gate; None
                disables it.
            distance: optional precomputed EDT, same shape as ``mask``
                (e.g. cropped from a ``ComputeEDT`` output covering the same
                roi). Values outside the mask are never read, so it's fine
                if they reflect neighboring objects rather than 0/background.
                None (default) computes it locally via ``edt.edt(mask, ...)``.
            neck_radius_mode: "fixed" (default) uses ``neck_radius_voxels``
                as-is. "adaptive" instead derives a per-object threshold via
                Otsu thresholding of this object's own sampled EDT radii
                (see ``_adaptive_neck_radius_nm``), floored at
                ``neck_radius_voxels`` so a degenerate/near-zero threshold
                on an object with no real bimodality can't disable the gate.

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


def _adaptive_neck_radius_nm(radii_nm, floor_nm):
    """Otsu-threshold an object's own sampled EDT radii to get a per-object
    neck-radius cut, floored at ``floor_nm`` (the caller's configured
    ``neck_radius_nm``).

    Experimental/baseline: this is intentionally the simplest version of
    "adaptive" (see docs/split_narrow_bridges_plan.md), meant to be treated
    as a starting point to experiment against, not a final answer. Falls
    back to the floor when there are too few samples or no real bimodality
    (a uniform blob's radii are all ~equal, and Otsu on a degenerate/near-
    constant histogram can return a near-zero threshold that would disable
    the gate entirely rather than adapt it).
    """
    radii_nm = np.asarray(radii_nm, dtype=float)
    if radii_nm.size < 2 or np.all(radii_nm == radii_nm.flat[0]):
        return float(floor_nm)
    try:
        otsu = threshold_otsu(radii_nm)
    except ValueError:
        return float(floor_nm)
    return max(float(otsu), float(floor_nm))


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
        self,
        mask,
        voxel_size,
        neck_radius_voxels,
        minimum_subregion_volume_voxels,
        distance=None,
        neck_radius_mode="fixed",
    ):
        if not np.any(mask):
            return np.zeros(mask.shape, dtype=np.uint8)

        if distance is None:
            distance = edt_module.edt(mask, anisotropy=tuple(voxel_size))

        # neck_radius_voxels is an isotropic voxel-count approximation
        # (neck_radius_nm / min(voxel_size)); round-trip it back to physical
        # nm to compare against the EDT-based (physical) radii. In
        # "adaptive" mode, Otsu-threshold this object's own EDT values
        # (already computed, no extra work) instead, floored at the fixed
        # value -- resolved before peak-finding so it also widens seed
        # spacing at the object's own natural scale, not just the later
        # boundary-thinness gate. (Sampling only at the watershed peaks
        # themselves doesn't work here -- peaks are local *maxima*, so they
        # never include the low-radius bridge voxels needed for Otsu to see
        # any bimodality at all.)
        floor_nm = neck_radius_voxels * min(voxel_size)
        if neck_radius_mode == "adaptive":
            neck_radius_nm = _adaptive_neck_radius_nm(distance[mask], floor_nm)
        else:
            neck_radius_nm = floor_nm
        resolved_neck_radius_voxels = neck_radius_nm / min(voxel_size)

        min_distance = max(1, int(round(resolved_neck_radius_voxels)))
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
        self,
        mask,
        voxel_size,
        neck_radius_voxels,
        minimum_subregion_volume_voxels,
        distance=None,
        neck_radius_mode="fixed",
    ):
        if not np.any(mask):
            return np.zeros(mask.shape, dtype=np.uint8)

        if distance is None:
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
        # (physical) node radii. In "adaptive" mode, Otsu-threshold the
        # skeleton-node radii already sampled above instead, floored at the
        # fixed value.
        floor_nm = neck_radius_voxels * min(voxel_size)
        if neck_radius_mode == "adaptive":
            neck_radius_nm = _adaptive_neck_radius_nm(radii, floor_nm)
        else:
            neck_radius_nm = floor_nm

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
        unassigned = mask & (node_assignment == 0)
        if np.any(unassigned):
            # Rare on real, jagged/noisy segmentation: some mask voxels
            # aren't reachable from any skeleton marker at watershed's
            # default connectivity (e.g. a bridge that's only diagonally
            # connected). Retry with full connectivity before giving up --
            # this recovers the common case cheaply (one extra watershed
            # call only when needed).
            node_assignment = watershed(
                np.zeros(mask.shape), markers, mask=mask, connectivity=mask.ndim
            )
            unassigned = mask & (node_assignment == 0)
            if np.any(unassigned):
                # Even full connectivity couldn't reach every voxel -- most
                # likely a genuinely separate sub-component sharing this
                # object's mask/id as a data artifact, not a bug in the
                # flood itself. Leave these unclaimed (mask_coords below
                # excludes them) rather than crashing bincount on a -1
                # index: the caller only overwrites voxels with piece > 0,
                # so unclaimed voxels simply keep their original label.
                logger.warning(
                    "%d/%d mask voxels unreachable from any skeleton node "
                    "even at full connectivity -- leaving unclaimed (keeps "
                    "original label) instead of crashing.",
                    int(unassigned.sum()), int(mask.sum()),
                )
        mask_coords = np.argwhere(mask & (node_assignment > 0))
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


def _principal_axis(points):
    """Principal axis (unit vector, sign arbitrary) of a point cloud, plus its
    linearity ``l1 / (l1 + l2 + l3)`` -- 1 for a perfect line, ~1/3 for a blob.
    Linearity is what lets the direction test recuse itself: the "does the
    bridge follow the tube's own axis" question is meaningless when the thing
    on either side isn't tube-shaped to begin with."""
    p = np.asarray(points, dtype=float)
    p = p - p.mean(axis=0)
    if len(p) < 3:
        return np.array([1.0, 0.0, 0.0]), 0.0
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(p.T))
    total = max(eigenvalues.sum(), 1e-12)
    return eigenvectors[:, -1], float(eigenvalues[-1] / total)


def _grow_sides(g, segment, original_neighbors, positions, fit_nm):
    """Flood outward from a candidate cut on each side, blocked by the segment
    itself, out to ``fit_nm`` of arclength. Returns the node lists of the two
    largest sides (a cut with fewer than two sides isn't a cut).

    Bounded by arclength rather than run to completion on purpose: we want each
    side's *local* axis at the junction, not the average direction of an entire
    branched network hanging off it a micron away.
    """
    segment_set = set(segment)
    groups, claimed = [], set()
    for entry in {
        nb for n in segment_set for nb in original_neighbors[n] if nb not in segment_set
    }:
        if entry in claimed:
            continue
        nodes, visited, frontier = [], set(segment_set) | {entry}, [(entry, 0.0)]
        while frontier:
            current, arc = frontier.pop(0)
            if arc > fit_nm:
                continue
            nodes.append(current)
            claimed.add(current)
            for nb in original_neighbors[current]:
                if nb not in visited:
                    visited.add(nb)
                    frontier.append((nb, arc + g[current][nb]["weight"]))
        if nodes:
            groups.append(nodes)
    groups.sort(key=len, reverse=True)
    return groups[:2]


class TubeDirectionSplit(SkeletonGraphSplit):
    """Skeleton splitting for *tubular* objects (mitochondria), where the
    radius-only reasoning both other strategies rely on breaks down.

    Three changes from SkeletonGraphSplit, each forced by a measured failure on
    a real object (id 1278, jrc_axolotl-heart-1 mito, 6.4um crop -- see
    docs/split_narrow_bridges_plan.md):

    1. **Simultaneous cutting.** SkeletonGraphSplit removes one thin segment at
       a time and keeps the cut only if that single removal disconnects the
       graph. A field of packed mitochondria merges into a *mesh*, not a tree
       (this object's skeleton has 2117 independent cycles: A touches B touches
       C touches A), so cutting any one bridge leaves the rest holding it
       together -- 380 candidate bridges, only 35 of which disconnect anything
       on their own. All qualifying segments are removed together and the
       components taken afterwards, which recovered 157 pieces where the
       one-at-a-time rule found ~35. This also removes the per-candidate whole
       graph copy, which cost 4 minutes on a 19k-node skeleton.

    2. **Prominence gate** (``prominence_max``): ``r_neck / r_flank``, a
       scale-free replacement for asking whether ``r_neck`` clears an absolute
       nm threshold. A uniformly thin tubule has prominence ~1 at every point
       and never qualifies no matter how far below ``neck_radius_nm`` it sits.

    3. **Direction gate** -- the one that actually does the work here. Compare
       each side's own principal axis to the direction of the bridge between
       them::

           axis_align   = |a_L . a_R|                  1 => the tubes are parallel
           bridge_along = max(|a_L . b|, |a_R . b|)    1 => bridge runs along their axis

       ``bridge_along`` high *and* ``axis_align`` high means one tube narrowing
       and continuing straight through -- left joined. Low ``bridge_along``
       with high ``axis_align`` is two parallel tubes joined by a perpendicular
       stub (the classic side-by-side merge); low ``axis_align`` is two tubes
       crossing. Both get cut.

    Why direction rather than more radius statistics: measured on the crop
    above, prominence is *identical* between genuine single-tube continuations
    (median 0.14) and real side-by-side merges (0.12) at a tight threshold --
    it has no discriminating power there, because everything that thin is
    already a one-voxel pinch. The two gates are complementary rather than
    redundant: at a loose threshold prominence separates cleanly (0.74 vs 0.13)
    and rescues the bad threshold, and together they hold the piece count flat
    (82/81/81/71 substantial pieces) across neck_radius_nm = 45..100nm, where
    the radius test alone swings 157 -> 514. That threshold-independence is the
    point -- it is the open question in the design doc, answered.
    """

    def __init__(
        self,
        prominence_max=0.35,
        direction_fit_nm=600.0,
        linearity_min=0.6,
        bridge_along_min=0.8,
        axis_align_min=0.8,
        bridge_along_max_merge=0.5,
        axis_align_max_cross=0.5,
        require_merge_evidence=True,
    ):
        """require_merge_evidence: which way the direction test defaults when
        it can't classify a junction.

        True (default) cuts ONLY on positive evidence of a merge -- both sides
        tube-like AND either parallel-with-perpendicular-bridge (side-by-side)
        or non-parallel (crossing). Anything ambiguous or blobby stays joined.

        False cuts unless the junction is positively a straight-through
        continuation. That was the first version and it over-splits badly: on
        real data half the candidates (88 ambiguous + 72 non-tube-like out of
        322) fall in neither class, and defaulting those to "cut" carves lobes
        and dents off otherwise-fine mitochondria. Kept only for comparison --
        over-splitting is the failure mode this whole project exists to avoid,
        so the burden of proof belongs on the cut, not on the join.
        """
        self.prominence_max = float(prominence_max)
        self.direction_fit_nm = float(direction_fit_nm)
        self.linearity_min = float(linearity_min)
        self.bridge_along_min = float(bridge_along_min)
        self.axis_align_min = float(axis_align_min)
        self.bridge_along_max_merge = float(bridge_along_max_merge)
        self.axis_align_max_cross = float(axis_align_max_cross)
        self.require_merge_evidence = bool(require_merge_evidence)

    def find_subpieces(
        self,
        mask,
        voxel_size,
        neck_radius_voxels,
        minimum_subregion_volume_voxels,
        distance=None,
        neck_radius_mode="fixed",
    ):
        if not np.any(mask):
            return np.zeros(mask.shape, dtype=np.uint8)

        if distance is None:
            distance = edt_module.edt(mask, anisotropy=tuple(voxel_size))
        skel = skeletonize(mask)
        if not np.any(skel):
            return mask.astype(np.uint8)

        skel_coords = np.argwhere(skel)
        radii = distance[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]]

        skeleton = skimage_to_custom_skeleton_fast(skel, spacing=voxel_size)
        skeleton.radii = list(radii)
        g = skeleton.skeleton_to_graph()
        num_nodes = g.number_of_nodes()
        if num_nodes <= 1:
            return mask.astype(np.uint8)

        floor_nm = neck_radius_voxels * min(voxel_size)
        if neck_radius_mode == "adaptive":
            neck_radius_nm = _adaptive_neck_radius_nm(radii, floor_nm)
        else:
            neck_radius_nm = floor_nm

        node_assignment, mask_coords, nearest_node_idx, node_voxel_counts = (
            self._assign_voxels_to_nodes(mask, skel_coords, num_nodes)
        )

        original_neighbors = {n: list(g.neighbors(n)) for n in g.nodes}
        positions = {
            n: np.asarray(g.nodes[n]["position_nm"], dtype=float) for n in g.nodes
        }

        thin_nodes = {n for n in g.nodes if g.nodes[n]["radius"] < neck_radius_nm}
        thin_components = list(nx.connected_components(g.subgraph(thin_nodes)))
        candidate_segments = [
            segment
            for segment in thin_components
            if not any(len(original_neighbors[n]) < 2 for n in segment)
        ]

        accepted = []
        for segment in candidate_segments:
            sides = _grow_sides(
                g, segment, original_neighbors, positions, self.direction_fit_nm
            )
            if len(sides) < 2 or min(len(s) for s in sides) < 3:
                continue

            r_neck = min(g.nodes[n]["radius"] for n in segment)
            r_flank = max(max(g.nodes[n]["radius"] for n in s) for s in sides)
            if r_neck / max(r_flank, 1e-6) >= self.prominence_max:
                continue  # dip too shallow relative to what it joins

            left = np.array([positions[n] for n in sides[0]])
            right = np.array([positions[n] for n in sides[1]])
            axis_left, linearity_left = _principal_axis(left)
            axis_right, linearity_right = _principal_axis(right)
            bridge = right.mean(axis=0) - left.mean(axis=0)
            bridge_norm = np.linalg.norm(bridge)
            if bridge_norm <= 1e-6:
                # Degenerate: no usable bridge direction, so no evidence either way.
                if self.require_merge_evidence:
                    continue
                accepted.append(segment)
                continue

            bridge /= bridge_norm
            bridge_along = max(
                abs(float(axis_left @ bridge)), abs(float(axis_right @ bridge))
            )
            axis_align = abs(float(axis_left @ axis_right))
            tube_like = min(linearity_left, linearity_right) > self.linearity_min

            is_continuation = (
                tube_like
                and bridge_along > self.bridge_along_min
                and axis_align > self.axis_align_min
            )
            is_side_by_side = (
                tube_like
                and axis_align > self.axis_align_min
                and bridge_along < self.bridge_along_max_merge
            )
            is_crossing = tube_like and axis_align < self.axis_align_max_cross

            if self.require_merge_evidence:
                if not (is_side_by_side or is_crossing):
                    continue  # ambiguous or blobby -> leave joined
            elif is_continuation:
                continue
            accepted.append(segment)

        if not accepted:
            return mask.astype(np.uint8)

        # Simultaneous removal -- see the class docstring for why one-at-a-time
        # cannot separate a cyclic merge mesh.
        h = g.copy()
        for segment in accepted:
            h.remove_nodes_from(segment)
        components = list(nx.connected_components(h))
        if len(components) <= 1:
            return mask.astype(np.uint8)

        if minimum_subregion_volume_voxels:
            components = self._dissolve_small_components(
                components, node_voxel_counts, original_neighbors,
                minimum_subregion_volume_voxels,
            )
            if len(components) <= 1:
                return mask.astype(np.uint8)

        node_to_piece = np.zeros(num_nodes, dtype=np.int32)
        for piece_index, component in enumerate(components, start=1):
            node_to_piece[list(component)] = piece_index
        self._resolve_cut_nodes(node_to_piece, original_neighbors, num_nodes)

        labels = np.zeros(mask.shape, dtype=np.int32)
        labels[mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2]] = node_to_piece[
            nearest_node_idx
        ]
        return labels

    @staticmethod
    def _assign_voxels_to_nodes(mask, skel_coords, num_nodes):
        """Geodesic nearest-skeleton-node assignment (see SkeletonGraphSplit
        for why a straight-line nearest neighbour is wrong here)."""
        markers = np.zeros(mask.shape, dtype=np.int32)
        markers[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]] = np.arange(
            1, num_nodes + 1
        )
        node_assignment = watershed(np.zeros(mask.shape, np.float32), markers, mask=mask)
        if np.any(mask & (node_assignment == 0)):
            node_assignment = watershed(
                np.zeros(mask.shape, np.float32), markers, mask=mask,
                connectivity=mask.ndim,
            )
        mask_coords = np.argwhere(mask & (node_assignment > 0))
        nearest_node_idx = (
            node_assignment[mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2]] - 1
        )
        node_voxel_counts = np.bincount(nearest_node_idx, minlength=num_nodes)
        return node_assignment, mask_coords, nearest_node_idx, node_voxel_counts

    @staticmethod
    def _dissolve_small_components(
        components, node_voxel_counts, original_neighbors, min_volume_voxels
    ):
        """Merge undersized pieces back into the neighbour they were cut from.

        Done on the component graph rather than the voxel array (the label-array
        version in ``_merge_small_fragments`` costs a full-array dilation per
        merge, which is fine for a handful of fragments and not for the hundreds
        a packed mito field produces).
        """
        node_component = {}
        for i, component in enumerate(components):
            for n in component:
                node_component[n] = i
        volumes = [int(node_voxel_counts[list(c)].sum()) for c in components]

        adjacency = {i: set() for i in range(len(components))}
        for n, neighbors in original_neighbors.items():
            a = node_component.get(n)
            if a is None:  # a cut-out node: joins whatever it touched
                touched = {
                    node_component[nb] for nb in neighbors if nb in node_component
                }
                for x in touched:
                    adjacency[x] |= touched - {x}
                continue
            for nb in neighbors:
                b = node_component.get(nb)
                if b is not None and b != a:
                    adjacency[a].add(b)
                    adjacency[b].add(a)

        parent = list(range(len(components)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        while True:
            roots = {find(i) for i in range(len(components))}
            if len(roots) <= 1:
                break
            small = [r for r in roots if volumes[r] < min_volume_voxels]
            if not small:
                break
            worst = min(small, key=lambda r: volumes[r])
            options = {find(x) for x in adjacency[worst]} - {worst}
            if not options:
                break
            target = max(options, key=lambda r: volumes[r])
            parent[worst] = target
            volumes[target] += volumes[worst]
            adjacency[target] |= adjacency[worst] - {target}

        merged = {}
        for i, component in enumerate(components):
            merged.setdefault(find(i), set()).update(component)
        return list(merged.values())

    @staticmethod
    def _resolve_cut_nodes(node_to_piece, original_neighbors, num_nodes):
        """Cut-out neck nodes have no piece; give each one its nearest surviving
        neighbour's. A handful of voxels at the pinch point either way."""
        for n in range(num_nodes):
            if node_to_piece[n] != 0:
                continue
            visited, queue = {n}, list(original_neighbors[n])
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                if node_to_piece[current] != 0:
                    node_to_piece[n] = node_to_piece[current]
                    break
                queue.extend(original_neighbors[current])


_STRATEGY_REGISTRY = {
    "edt_watershed": EDTWatershedSplit,
    "skeleton_graph": SkeletonGraphSplit,
    "tube_direction": TubeDirectionSplit,
}


class SplitNarrowBridges(ComputeConfigMixin):
    def __init__(
        self,
        segmentation_path,
        output_path,
        strategy="edt_watershed",
        neck_radius_nm=0,
        neck_radius_mode="fixed",
        minimum_subregion_volume_nm_3=None,
        minimum_object_volume_to_split_nm_3=0,
        max_pieces_per_object=1024,
        csv_path=None,
        edt_path=None,
        precompute_edt=False,
        edt_padding_nm=None,
        minimum_volume_nm_3=0,
        maximum_volume_nm_3=np.inf,
        num_workers=10,
        timeout=5,
        concurrency_limit=None,
        chunk_shape=None,
        delete_tmp=True,
        retry_on_oom=True,
        memory_retry_max=3,
        peak_bytes_baseline=250_000_000,
        peak_bytes_per_voxel=17.0,
        memory_safety_multiplier=2.0,
        memory_fraction=0.60,
        subpieces_cache_max_bytes=None,
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
                In "adaptive" ``neck_radius_mode``, this instead acts as a
                floor -- see ``neck_radius_mode``.
            neck_radius_mode: "fixed" (default) uses ``neck_radius_nm``
                as-is for every object. "adaptive" instead derives a
                per-object threshold via Otsu thresholding of that object's
                own sampled EDT radii (see ``_adaptive_neck_radius_nm``),
                floored at ``neck_radius_nm`` -- experimental/baseline,
                intended as a starting point to experiment against (a small
                vs. huge nucleus don't share a natural scale, so a single
                fixed value across all objects of a class may be wrong).
            minimum_subregion_volume_nm_3: Optional accept-gate -- a
                candidate split is only kept if the resulting subregions
                (after merging any that fall below this volume into their
                largest neighbor) still number >= 2. None (default) derives
                this from ``minimum_volume_nm_3`` instead of disabling the
                gate: a split producing a piece below the final volume
                filter would just get deleted (voxels dropped, not
                re-merged) by the deferred ``CleanConnectedComponents`` pass
                anyway, so it's rejected up front here instead, keeping
                those voxels attached to the parent object. Pass 0
                explicitly to fully disable the gate regardless of
                ``minimum_volume_nm_3``; pass a value to use a different
                threshold than the final filter.
            minimum_object_volume_to_split_nm_3: Optional pre-filter, checked
                against each object's *total* volume (from ``csv_path``'s
                "Volume (nm^3)" column) before any mask is even read --
                objects below this are never handed to ``strategy`` and are
                written through unchanged. Default 0 disables it (every
                object is a split candidate, the prior behavior). Distinct
                from ``minimum_subregion_volume_nm_3``, which gates whether
                one particular candidate *cut* is accepted once splitting is
                already underway -- this instead skips the attempt
                entirely, for objects too small overall to plausibly be two
                real merged things (e.g. segmentation-noise blobs, where any
                "neck" found is more likely a shape artifact than a genuine
                merge, and letting it split just fragments noise into more
                noise).
            max_pieces_per_object: Safety cap on how many subpieces a single
                object may split into; exceeding it raises rather than
                silently truncating. Also doubles as the fixed stride
                reserved for each split object's new IDs (``new_id_base =
                max_original_id + 1 + rank * max_pieces_per_object``), so it
                can't be disabled outright (``None``) without reworking ID
                assignment -- it can only be raised. Default raised from 64
                to 1024 (was too tight for real data: real id 4665 in
                jrc_mus-cerebellum-2, plausibly a segmentation-noise blob
                rather than a real merged-nuclei chain, split into 122
                pieces -- see docs/split_narrow_bridges_plan.md). Still a
                real safety net against genuinely pathological runaway
                fragmentation (thousands of pieces), just not tuned tightly
                for now.
            csv_path: Optional path to a CSV with per-object bounding boxes
                (the kind Measure produces). If None, Measure is run on the
                segmentation to generate one at
                ``<output_path>/bboxes/<leaf>.csv``.
            edt_path: Optional path to a precomputed EDT dataset (e.g. from
                ``ComputeEDT`` run over ``segmentation_path``). When given,
                each object reads its EDT window from here instead of
                recomputing ``edt.edt(mask, ...)`` from scratch -- the
                expensive part for very large merged objects. Must cover the
                same voxel grid as ``segmentation_path`` (same voxel size and
                total roi); a per-object shape mismatch raises rather than
                silently misaligning.
            precompute_edt: Only consulted when ``edt_path`` is None. False
                (default) computes ``edt.edt(mask, ...)`` per object, inside
                the same memory-aware per-object wave dispatch
                ``split_objects()`` already uses (see ``_estimate_peak_bytes``)
                -- each object's crop is tightly padded to its own real
                extent, so this scales safely with no separate memory
                budgeting needed. Set to True to instead run ``ComputeEDT``
                once over the whole ``segmentation_path`` up front (into a
                scratch dataset), the same way ``csv_path=None`` auto-runs
                ``Measure`` -- this amortizes the EDT cost across objects
                (worth it if you're iterating on thresholds across many
                repeated runs over the same segmentation), but ``ComputeEDT``
                dispatches on a fixed block grid with no memory-aware wave
                planning of its own: every block is padded by
                ``edt_padding_nm``/``neck_radius_nm`` regardless of whether
                it's actually near a real object boundary, which can push a
                block's peak memory past whatever's configured per dask
                slot (this is what failed on jrc_mus-cerebellum-3 in
                production -- see docs/split_narrow_bridges_plan.md). Prefer
                explicitly running ``compute-edt`` once yourself and passing
                the result via ``edt_path`` (reused across many runs) over
                setting ``precompute_edt=True`` here, if you do need the
                amortization.
            edt_padding_nm: Padding (nm) for the auto-computed EDT dataset
                when ``precompute_edt`` runs; ignored otherwise. Defaults to
                ``max(neck_radius_nm, one voxel)`` -- windowed EDT can only
                overestimate the true distance, so this only needs to
                comfortably exceed ``neck_radius_nm`` for the "is this thin?"
                decision to be trustworthy, not the physical scale of any
                object (see ``ComputeEDT``'s docstring).
            minimum_volume_nm_3, maximum_volume_nm_3: Optional final
                dataset-level volume filter, applied via
                ``CleanConnectedComponents`` after splitting/writing is
                complete (defaults 0/inf disable it). Distinct from
                ``minimum_subregion_volume_nm_3``, which only gates whether
                an individual candidate split is accepted -- this instead
                prunes small/oversized *final* objects (including
                un-split ones) from the output dataset, mirroring the
                deferred-filtering pattern ``MutexWatershed`` uses with
                ``do_opening`` (split first with lenient/no volume limits,
                then filter the fully-formed result) so overmerged blobs
                get a chance to split before max-volume filtering would
                otherwise drop them, and spurious over-split fragments get
                caught by min-volume filtering afterward.
            num_workers: Number of parallel dask workers.
            timeout: Base timeout (seconds) for ImageDataInterface reads,
                scaled up per ``ImageDataInterface.read_with_retries``
                (``timeout * attempt``, up to 10 attempts) and, when
                ``concurrency_limit`` auto-resolves above 1 (see below),
                further scaled by the biggest object's estimated chunk
                count so a single attempt has enough wall-clock to finish
                reading a huge bbox.
            concurrency_limit: tensorstore ``data_copy_concurrency`` /
                ``file_io_concurrency`` limit for the segmentation/EDT
                reads. None (default) auto-resolves: this codebase pins
                ``cores == processes`` and forces every numeric library to
                1 thread (see ``dask_util.start_dask``'s
                ``job_script_prologue``) specifically so process count is
                the only lever on CPU use in the normal multi-worker wave
                dispatch (``num_workers > 1``) -- there, auto-resolve keeps
                the safe ``1`` (matches the prior hardcoded
                ``ImageDataInterface`` default; unbounded concurrency per
                worker process would let tensorstore oversubscribe cores
                shared with sibling worker processes on the same node).
                When ``num_workers <= 1`` (the synchronous, no-cluster path
                -- see ``dask_util.start_dask``'s early return -- e.g. a
                one-off script or a solo high-memory validation run),
                there's no sibling process to oversubscribe against, so
                auto-resolve instead uses every core available to this
                process (``os.sched_getaffinity``, respecting cgroup/LSF/
                Slurm pinning). This is exactly the fix a real id-4665
                (``jrc_mus-cerebellum-2``) validation needed: the default
                ``concurrency_limit=1`` read its ~23,000-chunk bbox one
                chunk at a time over network storage and timed out
                entirely before any memory pressure began; raising
                concurrency (and timeout) let the read finish in under a
                second. Pass an explicit int to override auto-resolution
                either way.
                For real lsf/slurm/sge wave dispatch (``num_workers > 1``),
                the initial ``1`` above is only the value baked into this
                instance's IDIs at construction; ``split_objects`` also
                rescales it per-wave inside each worker (see
                ``split_id``'s ``processes_per_job``) to that worker's fair
                share of its *job's* real CPU affinity -- a job's cpuset is
                shared by every dask worker process running inside it
                (affinity isn't divided per-process), so a solo-item
                high-memory wave (``processes=1``) still gets its job's
                full affinity count instead of being stuck at ``1``, while
                a multi-process wave divides that same count across its
                siblings rather than each one claiming the whole thing.
            delete_tmp: Delete the per-object scratch directory when done.
            retry_on_oom: Halve processes-per-slot and retry on worker OOM
                (see ``dask_util.run_with_oom_retry``).
            memory_retry_max: Max OOM-driven retries before raising.
            peak_bytes_baseline, peak_bytes_per_voxel: Estimator constants
                for per-object peak RSS, used to group objects into
                memory-aware dask waves (mirrors ``Skeletonize``). Peak is
                ``mask (bool) + distance (float64) + markers (int32) +
                watershed labels (int32)`` = 1 + 8 + 4 + 4 = 17 bytes/voxel
                when no EDT is precomputed (``edt_idi`` is None, so
                ``find_subpieces`` calls ``edt.edt(mask, ...)`` itself,
                needing its own float64 distance array); when an EDT is
                precomputed, 8 bytes/voxel are subtracted (no local
                ``edt.edt`` call) leaving ~9 bytes/voxel.
            memory_safety_multiplier: Multiplier applied to the per-object
                peak estimate before wave planning, absorbing variance in
                how many markers/pieces a given object splits into.
            memory_fraction: Fraction of per-slot memory considered usable
                when planning waves (rest is dask/OS/library overhead).
            subpieces_cache_max_bytes: Byte budget for the write phase's
                per-worker-process cache of split objects' full ``subpieces``
                arrays (see ``_load_subpieces_cached``). Loading a split
                object's array is only needed once per worker no matter how
                many of its (often many) blocks that worker happens to
                process -- caching it avoids re-reading the same file from
                disk repeatedly -- but with no bound, that cache grows for
                the life of the worker process and a single pathologically
                large split object (or several moderately large ones
                accumulating together) can exceed the worker's own
                ``memory_limit`` and get it killed. None (default) derives a
                budget from the loaded dask-config's per-worker memory (half
                of it), falling back to 4GiB if no jobqueue config is
                available (e.g. synchronous runs).
        """
        super().__init__(num_workers)
        self.concurrency_limit = dask_util.resolve_concurrency_limit(
            num_workers, concurrency_limit
        )
        self.segmentation_path = segmentation_path
        self.segmentation_idi = ImageDataInterface(
            segmentation_path,
            timeout=timeout,
            chunk_shape=chunk_shape,
            concurrency_limit=self.concurrency_limit,
        )
        self.output_path = str(output_path).rstrip("/")
        self.roi = self.segmentation_idi.roi

        # Per-instance suffix so concurrent runs sharing output_path don't
        # collide on the scratch/merge dirs. Needed early: also names the
        # auto-computed EDT scratch dataset below.
        self._run_id = uuid.uuid4().hex[:8]

        self._edt_scratch_path = None
        self.edt_idi = None
        if edt_path is not None:
            self.edt_idi = ImageDataInterface(
                edt_path,
                timeout=timeout,
                chunk_shape=chunk_shape,
                concurrency_limit=self.concurrency_limit,
            )
        elif precompute_edt:
            from cellmap_analyze.process.compute_edt import ComputeEDT

            padding_nm = edt_padding_nm
            if padding_nm is None:
                padding_nm = max(
                    float(neck_radius_nm), min(self.segmentation_idi.original_voxel_size)
                )
            self._edt_scratch_path = get_output_path_from_input_path(
                self.output_path, f"_edt_scratch_{self._run_id}"
            )
            logger.info(
                "precompute_edt=True and no edt_path given; running "
                "ComputeEDT over %s once (padding_nm=%.4g) -> %s, so every "
                "object reuses it instead of recomputing edt.edt(mask, ...) "
                "itself.",
                self.segmentation_path,
                padding_nm,
                self._edt_scratch_path,
            )
            ComputeEDT(
                segmentation_path=self.segmentation_path,
                output_path=self._edt_scratch_path,
                padding_nm=padding_nm,
                num_workers=num_workers,
                chunk_shape=chunk_shape,
            ).calculate_edt()
            self.edt_idi = ImageDataInterface(
                f"{self._edt_scratch_path}/s0",
                timeout=timeout,
                chunk_shape=chunk_shape,
                concurrency_limit=self.concurrency_limit,
            )

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

        self.minimum_object_volume_to_split_nm_3 = float(
            minimum_object_volume_to_split_nm_3
        )
        if self.minimum_object_volume_to_split_nm_3 > 0 and self.ids:
            too_small = (
                self.bbox_df.loc[self.ids, "Volume (nm^3)"]
                < self.minimum_object_volume_to_split_nm_3
            )
            num_skipped = int(too_small.sum())
            if num_skipped:
                logger.info(
                    "Skipping split attempts for %d/%d objects below "
                    "minimum_object_volume_to_split_nm_3=%.4g nm^3 (written "
                    "through unchanged).",
                    num_skipped,
                    len(self.ids),
                    self.minimum_object_volume_to_split_nm_3,
                )
                self.ids = too_small.index[~too_small].tolist()

        if neck_radius_mode not in ("fixed", "adaptive"):
            raise ValueError(
                f"Unknown neck_radius_mode {neck_radius_mode!r}; valid: "
                f"'fixed', 'adaptive'"
            )
        self.neck_radius_mode = neck_radius_mode

        original_voxel_size = self.segmentation_idi.original_voxel_size
        voxel_volume = float(np.prod(original_voxel_size))
        self.neck_radius_voxels = float(neck_radius_nm) / min(original_voxel_size)
        if minimum_subregion_volume_nm_3 is None:
            # Default: derive from the final minimum_volume_nm_3 filter. A
            # split producing a piece below that threshold would just get
            # deleted -- voxels dropped, not merged back -- by the final
            # CleanConnectedComponents pass anyway, so reject the split up
            # front instead and keep those voxels attached to the parent
            # object. When minimum_volume_nm_3 is also left at its default
            # (0), this derives to 0, which is falsy below -- identical to
            # the previous None/disabled default.
            minimum_subregion_volume_nm_3 = minimum_volume_nm_3
        self.minimum_subregion_volume_voxels = (
            float(minimum_subregion_volume_nm_3) / voxel_volume
            if minimum_subregion_volume_nm_3
            else None
        )

        self.max_pieces_per_object = int(max_pieces_per_object)
        self.minimum_volume_nm_3 = float(minimum_volume_nm_3)
        self.maximum_volume_nm_3 = float(maximum_volume_nm_3)
        self.delete_tmp = delete_tmp
        self.retry_on_oom = retry_on_oom
        self.memory_retry_max = memory_retry_max
        self.peak_bytes_baseline = float(peak_bytes_baseline)
        self.peak_bytes_per_voxel = float(peak_bytes_per_voxel)
        self.memory_safety_multiplier = float(memory_safety_multiplier)
        self.memory_fraction = float(memory_fraction)
        self.subpieces_cache_max_bytes = (
            int(subpieces_cache_max_bytes)
            if subpieces_cache_max_bytes is not None
            else None
        )

        if self.concurrency_limit > 1 and self.ids:
            # Only worth scaling in the same auto-resolved-concurrency
            # case above: reads there are chunk-count-bound (network
            # storage, one round trip per chunk group), so a single
            # timeout attempt needs headroom proportional to the biggest
            # object's chunk span, not just a fixed default sized for
            # small objects. Calibrated against the real id-4665 case:
            # ~23,000 chunks needed timeout~=120s (vs. the 5s default) to
            # complete a single attempt -- roughly chunks/1000, seconds.
            max_chunks = max(self._estimate_num_chunks(i) for i in self.ids)
            scaled_timeout = min(max(timeout, timeout * max_chunks / 1000.0), 600.0)
            self.segmentation_idi.timeout = scaled_timeout
            if self.edt_idi is not None:
                self.edt_idi.timeout = scaled_timeout

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
        voxel_size = np.array(segmentation_idi.voxel_size, dtype=float)
        sf = segmentation_idi.voxel_size_scale_factor
        padding = voxel_size  # 1 voxel in each direction

        # Measure's "MIN/MAX (nm)" columns are voxel-*center* coordinates
        # (see measure_util.py), not edges. Naively doing `MIN*sf - padding`
        # / `MAX*sf + padding` looks symmetric but isn't: MIN*sf is already
        # half a voxel inside the object's true bounding box, so subtracting
        # a full voxel of padding correctly reaches a full voxel past the
        # low edge -- but MAX*sf is also half a voxel *inside* the box, so
        # adding a full voxel of padding only reaches the far edge of the
        # object's own last voxel, giving ~zero real margin on the high
        # side. Shifting by voxel_size/2 first lands both bounds exactly on
        # grid edges, so the padding added afterward is a genuine, symmetric
        # full voxel on every side. (voxel_size is cast to float above --
        # funlib.geometry.Coordinate does integer division, so `/2` on an
        # odd or unit voxel_size would silently truncate the shift to 0.)
        min_point = (
            np.array(
                [row["MIN Z (nm)"] * sf, row["MIN Y (nm)"] * sf, row["MIN X (nm)"] * sf]
            )
            - voxel_size / 2
        )
        max_point = (
            np.array(
                [row["MAX Z (nm)"] * sf, row["MAX Y (nm)"] * sf, row["MAX X (nm)"] * sf]
            )
            + voxel_size / 2
        )
        start_point = min_point - padding
        end_point = max_point + padding
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
        edt_idi: ImageDataInterface = None,
        neck_radius_mode: str = "fixed",
        processes_per_job: int = None,
    ):
        """Process a single object: extract its mask, find candidate
        subpieces, and (if split) persist the local labeling to a small
        per-object scratch file.

        Returns a dict describing the outcome; ``split_objects`` merges
        these on the driver before doing any blockwise writing.

        processes_per_job: When set (only meaningful for lsf/slurm/sge
            waves -- see ``split_objects``), rescales ``segmentation_idi``/
            ``edt_idi``'s ``concurrency_limit`` to this worker's fair share
            of its job's real CPU affinity: ``sched_getaffinity`` count //
            ``processes_per_job``. A single job's cpuset is shared by every
            dask worker *process* running inside it (cpuset affinity isn't
            divided per-process), so each sibling must independently claim
            only its share -- reading the raw affinity count directly (as
            ``SplitNarrowBridges.__init__``'s standalone/synchronous
            auto-resolution does) would let every worker in the job think
            it owns the whole job's cores, reintroducing the same
            oversubscription ``concurrency_limit`` exists to prevent, just
            one level down (job-shared instead of host-shared). None
            (default) leaves whatever ``concurrency_limit`` was set at
            construction untouched.
        """
        dask_util.rescale_idi_concurrency(
            (segmentation_idi, edt_idi), processes_per_job
        )

        roi = SplitNarrowBridges._object_roi(id_value, bbox_df, segmentation_idi)
        data = segmentation_idi.to_ndarray_ts(roi)
        mask = data == id_value
        del data  # never referenced again; drop the full-dtype array before
        # the EDT/markers/labels arrays get allocated below, so peak memory
        # doesn't carry a dead uint32/uint64-sized copy alongside them.
        if not np.any(mask):
            logger.warning(f"No voxels found for ID {id_value}, skipping")
            return {"id": int(id_value), "split": False}

        distance = None
        if edt_idi is not None:
            distance = edt_idi.to_ndarray_ts(roi)
            if distance.shape != mask.shape:
                raise ValueError(
                    f"Precomputed EDT shape {distance.shape} does not match "
                    f"segmentation crop shape {mask.shape} for object "
                    f"{id_value} (roi={roi}). edt_path must cover the same "
                    f"voxel grid as segmentation_path (e.g. produced by "
                    f"ComputeEDT run over this same segmentation)."
                )

        subpieces = strategy.find_subpieces(
            mask,
            segmentation_idi.original_voxel_size,
            neck_radius_voxels,
            minimum_subregion_volume_voxels,
            distance=distance,
            neck_radius_mode=neck_radius_mode,
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
    def _build_split_spatial_index(split_lookup):
        """Precompute a vectorizable bounds index for ``split_lookup`` once,
        driver-side, instead of re-deriving it per block.

        ``relabel_block_with_splits`` used to loop over every entry of
        ``split_lookup`` for *every* block (a Python-level
        ``Roi.intersect()`` call each time) to find the handful that
        actually overlap -- O(blocks * split objects). On a real production
        run (jrc_mus-cerebellum-2, 1507 split objects, ~1.74M blocks) this
        made the write phase run at ~2.8 blocks/sec (measured directly
        against the output store's growing file count) vs. ~22,000
        blocks/sec for the structurally similar ``ConnectedComponents``
        relabel pass over the same dataset shape/chunking -- on pace for
        multiple *days*, not the minutes every other phase took. Returning
        plain numpy bounds arrays here lets each block test overlap against
        every object in one vectorized comparison instead of a Python loop.
        """
        ids = list(split_lookup.keys())
        begins = np.array([split_lookup[i][2].begin for i in ids], dtype=np.float64)
        ends = np.array(
            [
                np.array(split_lookup[i][2].begin) + np.array(split_lookup[i][2].shape)
                for i in ids
            ],
            dtype=np.float64,
        )
        return ids, begins, ends

    @staticmethod
    def _load_subpieces_cached(scratch_path, cache, max_bytes):
        """Load a split object's scratch ``subpieces`` array, LRU-cached per
        worker *process* (byte-budgeted, not count-budgeted) for the
        lifetime of this write phase.

        Without this, a large split object's array gets reloaded from disk
        from scratch every single time any block overlapping its (often
        huge) bbox is processed -- for an object spanning thousands of
        blocks, that's thousands of redundant full re-reads of the same
        file. dask workers are long-lived processes reused across many
        block tasks, so a plain dict (passed in explicitly here rather than
        truly global, so it's easy to reason about/test) persists across
        those tasks within one worker -- but with no eviction, that cache
        grows unboundedly for the life of the worker process, and one
        pathologically large split object (or several moderate ones
        accumulating together) can exceed the worker's own memory_limit and
        get it killed (observed in production: a single ~5.8B-voxel object
        alongside a normal accumulation of smaller ones pushed a worker past
        its per-process limit, and since the scheduler just reassigns the
        same task to another similarly-loaded worker, it died the same way
        repeatedly). Evicting oldest-used entries first keeps the cache
        within a fixed byte budget while an object whose blocks are being
        processed back-to-back (the case the cache exists to help) stays
        hot via ``move_to_end``. A single object larger than the whole
        budget is still loaded and used for this call -- it's just not
        retained afterward, so it'll be re-read for its next block instead
        of wedging the cache permanently over budget.
        """
        subpieces = cache.get(scratch_path)
        if subpieces is not None:
            cache.move_to_end(scratch_path)
            return subpieces

        with np.load(scratch_path) as npz:
            subpieces = npz["subpieces"]

        incoming_bytes = subpieces.nbytes
        current_bytes = sum(arr.nbytes for arr in cache.values())
        while cache and current_bytes + incoming_bytes > max_bytes:
            _, evicted = cache.popitem(last=False)
            current_bytes -= evicted.nbytes
        if incoming_bytes <= max_bytes:
            cache[scratch_path] = subpieces
        return subpieces

    @staticmethod
    def relabel_block_with_splits(
        block_index,
        segmentation_idi: ImageDataInterface,
        output_idi: ImageDataInterface,
        split_lookup: dict,
        spatial_index,
        dtype,
        subpieces_cache_max_bytes,
    ):
        block = create_block_from_index(output_idi, block_index)
        data = segmentation_idi.to_ndarray_ts(block.write_roi).astype(dtype)

        ids, begins, ends = spatial_index
        if ids:
            block_begin = np.array(block.write_roi.begin, dtype=np.float64)
            block_end = block_begin + np.array(block.write_roi.shape, dtype=np.float64)
            # Cheap vectorized bbox-overlap test against every split object
            # at once; only the (usually zero, rarely more than a handful)
            # candidates that pass get the exact per-axis slicing below.
            candidates = np.nonzero(
                np.all((begins < block_end) & (ends > block_begin), axis=1)
            )[0]
        else:
            candidates = ()

        if len(candidates):
            voxel_size = output_idi.voxel_size
            cache = SplitNarrowBridges._subpieces_cache
            for idx in candidates:
                id_value = ids[idx]
                new_id_base, scratch_path, obj_roi = split_lookup[id_value]
                overlap = block.write_roi.intersect(obj_roi)
                if overlap.empty:
                    continue

                subpieces = SplitNarrowBridges._load_subpieces_cached(
                    scratch_path, cache, subpieces_cache_max_bytes
                )

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
                    region[piece_mask] = new_id_base + sub_region[piece_mask].astype(
                        dtype
                    )
                    data[data_slice] = region

        output_idi.ds[block.write_roi] = data

    # Per-worker-process cache for relabel_block_with_splits (see
    # _load_subpieces_cached) -- a plain class attribute so it persists
    # across many block tasks handled by the same long-lived dask worker
    # process, without needing a true module-global. OrderedDict for LRU
    # eviction (move_to_end/popitem(last=False)).
    _subpieces_cache = OrderedDict()

    def _resolve_subpieces_cache_max_bytes(self, config):
        """Byte budget for the write phase's per-worker subpieces cache (see
        ``_load_subpieces_cached``). Explicit constructor value wins;
        otherwise derive half of one worker's share of its job's memory from
        the loaded dask-config, falling back to a fixed 4GiB when no
        jobqueue config is available (e.g. synchronous/local runs)."""
        if self.subpieces_cache_max_bytes is not None:
            return self.subpieces_cache_max_bytes
        _, settings = dask_util._jobqueue_settings(config)
        job_memory = dask_util._job_memory_bytes(settings) if settings else None
        if job_memory:
            processes = int((settings or {}).get("processes", 1) or 1)
            return int(0.5 * job_memory / processes)
        return 4 * 1024**3

    def _write_output(self, split_lookup, dtype=None, output_path=None, base_config=None):
        if dtype is None:
            dtype = self.segmentation_idi.dtype
        if output_path is None:
            output_path = self.output_path

        output_idi = create_multiscale_dataset_idi(
            output_path,
            dtype=dtype,
            voxel_size=self.segmentation_idi.voxel_size,
            total_roi=self.roi,
            write_size=self.segmentation_idi.chunk_shape * self.segmentation_idi.voxel_size,
            original_voxel_size=self.segmentation_idi.original_voxel_size,
        )

        spatial_index = SplitNarrowBridges._build_split_spatial_index(split_lookup)
        subpieces_cache_max_bytes = self._resolve_subpieces_cache_max_bytes(base_config)

        num_blocks = dask_util.get_num_blocks(self.segmentation_idi, roi=self.roi)
        write_label = f"writing split output to {output_path}"

        def _phase(workers, config):
            return dask_util.compute_blockwise_partitions(
                num_blocks,
                workers,
                self.compute_args,
                logger,
                write_label,
                SplitNarrowBridges.relabel_block_with_splits,
                self.segmentation_idi,
                output_idi,
                split_lookup,
                spatial_index,
                dtype,
                subpieces_cache_max_bytes,
                config=config,
            )

        # Defense in depth alongside the cache budget above: if a worker
        # still dies (OOM from something else entirely), halve
        # processes-per-slot and retry rather than failing the whole write
        # outright -- the same safety net split-finding already gets.
        dask_util.run_with_oom_retry(
            _phase, self.num_workers, write_label, logger,
            max_retries=self.memory_retry_max,
            retry_on_oom=self.retry_on_oom,
            config=base_config,
        )

    def _bbox_iso_voxels(self, id_value):
        """Isotropic voxel count of ``id_value``'s (padded) bbox -- shared
        by the memory (``_estimate_peak_bytes``) and read-size
        (``_estimate_num_chunks``) estimators."""
        row = self.bbox_df.loc[id_value]
        voxel_size = self.segmentation_idi.voxel_size
        original_vs = self.segmentation_idi.original_voxel_size
        dx = (row["MAX X (nm)"] - row["MIN X (nm)"]) + 2 * voxel_size[0]
        dy = (row["MAX Y (nm)"] - row["MIN Y (nm)"]) + 2 * voxel_size[1]
        dz = (row["MAX Z (nm)"] - row["MIN Z (nm)"]) + 2 * voxel_size[2]
        min_vs = min(original_vs)
        return (dx * dy * dz) / (min_vs ** 3)

    def _estimate_peak_bytes(self, id_value):
        """Estimate per-object peak RSS from the cached bbox row.

        Uses the isotropic voxel count of the (padded) bbox times a
        per-voxel cost: ``mask (bool) + distance (float64) + markers
        (int32) + watershed labels (int32)`` = 17 bytes/voxel when no EDT
        is precomputed, or ~9 bytes/voxel when one is (no local
        ``edt.edt`` call, so no separate float64 distance array).
        ``memory_safety_multiplier`` absorbs cross-object variance (e.g.
        the number of watershed markers/pieces kept alive at once).
        """
        iso_voxels = self._bbox_iso_voxels(id_value)
        bytes_per_voxel = self.peak_bytes_per_voxel
        if self.edt_idi is not None:
            bytes_per_voxel -= 8.0
        peak = self.peak_bytes_baseline + bytes_per_voxel * iso_voxels
        return int(peak * self.memory_safety_multiplier)

    def _estimate_num_chunks(self, id_value):
        """Estimate how many storage chunks ``id_value``'s (padded) bbox
        spans -- read wall-clock scales with this, not with voxel count,
        since ``concurrency_limit`` gates chunk-level, not byte-level,
        parallelism."""
        chunk_voxels = float(np.prod(self.segmentation_idi.chunk_shape))
        return self._bbox_iso_voxels(id_value) / max(chunk_voxels, 1.0)

    def _log_wave_plan(self, waves):
        if not waves:
            return
        total_ids = sum(len(w.item_ids) for w in waves)
        biggest = max(w.max_estimated_peak_bytes for w in waves)
        logger.info(
            "Wave plan: %d wave(s) over %d objects (largest projected peak %.2f GB)",
            len(waves), total_ids, biggest / 1e9,
        )
        for i, wave in enumerate(waves, start=1):
            logger.info(
                "  wave %d/%d: processes/slot=%d, workers=%d, objects=%d, "
                "max projected peak %.2f GB",
                i, len(waves), wave.processes, wave.workers,
                len(wave.item_ids), wave.max_estimated_peak_bytes / 1e9,
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

        try:
            base_config = (
                dask_util._load_dask_config() if self.num_workers > 1 else None
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as e:
            logger.warning(
                "Could not load dask-config.yaml for wave planning (%s); "
                "running all objects in a single wave.",
                e,
            )
            base_config = None

        items = [(id_value, self._estimate_peak_bytes(id_value)) for id_value in self.ids]
        waves = dask_util.plan_memory_waves(
            items,
            self.num_workers,
            config=base_config,
            memory_fraction=self.memory_fraction,
        )
        self._log_wave_plan(waves)

        processes_per_job_by_wave = dask_util.wave_uses_shared_job_cpuset(base_config)

        results = []
        with io_util.TimingMessager("Finding candidate splits", logger):
            for wave_index, wave in enumerate(waves, start=1):
                wave_label = (
                    f"splitting narrow bridges for {self.segmentation_path} "
                    f"wave {wave_index}/{len(waves)} ({len(wave.item_ids)} objects)"
                )
                wave_ids = wave.item_ids
                wave_merge_dir = f"{tmp_merge_root}_wave{wave_index}"

                processes_per_job = (
                    wave.processes if processes_per_job_by_wave else None
                )

                def _wrapper(idx, _wave_ids=wave_ids, _procs=processes_per_job):
                    id_value = _wave_ids[idx]
                    return SplitNarrowBridges.split_id(
                        id_value,
                        self.segmentation_idi,
                        self.bbox_df,
                        self.strategy,
                        self.neck_radius_voxels,
                        self.minimum_subregion_volume_voxels,
                        self.max_pieces_per_object,
                        scratch_dir,
                        self.edt_idi,
                        neck_radius_mode=self.neck_radius_mode,
                        processes_per_job=_procs,
                    )

                def _phase(workers, config, _wrapper=_wrapper, _ids=wave_ids,
                           _merge=wave_merge_dir, _label=wave_label):
                    return dask_util.compute_blockwise_partitions(
                        len(_ids), workers, self.compute_args, logger, _label,
                        _wrapper,
                        merge_info=(SplitNarrowBridges._merge_split_results, _merge),
                        config=config,
                    )

                wave_results = dask_util.run_with_oom_retry(
                    _phase, wave.workers, wave_label, logger,
                    max_retries=self.memory_retry_max,
                    retry_on_oom=self.retry_on_oom,
                    config=wave.config,
                )
                results.extend(wave_results)

        split_results = [r for r in results if r["split"]]
        logger.info(f"{len(split_results)}/{len(self.ids)} objects split")

        apply_final_filter = (
            self.minimum_volume_nm_3 > 0 or self.maximum_volume_nm_3 < np.inf
        )
        write_path = (
            get_output_path_from_input_path(
                self.output_path, f"_unfiltered_{self._run_id}"
            )
            if apply_final_filter
            else self.output_path
        )

        try:
            if not split_results:
                self._write_output({}, output_path=write_path, base_config=base_config)
            else:
                split_results.sort(key=lambda r: r["id"])
                # Deliberately the full bbox_df index, not self.ids -- the
                # latter may have objects filtered out by
                # minimum_object_volume_to_split_nm_3 (never split, but
                # still written through with their original id), and a new
                # split id must never collide with one of those.
                max_original_id = (
                    int(self.bbox_df.index.max()) if len(self.bbox_df) else 0
                )
                split_lookup = {}
                for rank, r in enumerate(split_results):
                    new_id_base = max_original_id + 1 + rank * self.max_pieces_per_object
                    obj_roi = Roi(r["roi_begin"], r["roi_shape"])
                    split_lookup[r["id"]] = (new_id_base, r["scratch_path"], obj_roi)

                max_new_id = (
                    max_original_id + len(split_results) * self.max_pieces_per_object
                )
                new_dtype = np.min_scalar_type(max_new_id)

                self._write_output(
                    split_lookup, new_dtype, output_path=write_path, base_config=base_config
                )
        finally:
            if self.delete_tmp:
                shutil.rmtree(scratch_dir, ignore_errors=True)
                if self._edt_scratch_path is not None:
                    shutil.rmtree(self._edt_scratch_path, ignore_errors=True)

        if apply_final_filter:
            # Deferred-filtering pattern (see MutexWatershed's do_opening):
            # split first with no/lenient volume limits, then filter the
            # fully-formed result -- so overmerged blobs get a chance to
            # split before max-volume filtering would otherwise drop them.
            with io_util.TimingMessager(
                f"Applying final volume filter to {write_path}", logger
            ):
                CleanConnectedComponents(
                    input_path=write_path + "/s0",
                    output_path=self.output_path,
                    num_workers=self.num_workers,
                    minimum_volume_nm_3=self.minimum_volume_nm_3,
                    maximum_volume_nm_3=self.maximum_volume_nm_3,
                    delete_tmp=self.delete_tmp,
                ).clean_connected_components()
            if self.delete_tmp:
                dask_util.delete_tmp_dir_blockwise(
                    write_path + "/s0", self.num_workers, self.compute_args
                )

        logger.info("Splitting complete")

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
from skimage.morphology import skeletonize, binary_erosion
from tqdm import tqdm
import logging
import os
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


class Skeletonize(ComputeConfigMixin):
    def __init__(
        self,
        segmentation_path,
        output_path,
        csv_path,
        erosion=True,
        min_branch_length_nm=100,
        tolerance_nm=50,
        num_workers=10,
        timeout=5,
        sharded=True,
        shard_bits=1,
        minishard_bits=6,
        retry_on_oom=True,
        memory_retry_max=3,
        peak_bytes_baseline=250_000_000,
        peak_bytes_per_voxel=6.63,
        memory_fraction=0.60,
    ):
        """
        Skeletonize a segmentation, parallelized over IDs.

        Args:
            segmentation_path: Path to the segmentation zarr dataset
            output_path: Path to the output directory for skeletons
            csv_path: Path to CSV containing bounding box info with columns:
                     MIN X (nm), MIN Y (nm), MIN Z (nm), MAX X (nm), MAX Y (nm), MAX Z (nm)
                     and index column for IDs
            erosion: Controls pre-skeletonization erosion.
                     True or "full": standard binary erosion with 6-connectivity cross SE.
                     6: targeted removal of edge/vertex-only bridges (keep face-connected).
                     18: targeted removal of vertex-only bridges (keep face+edge-connected).
                     False or None: no erosion.
            min_branch_length_nm: Minimum branch length for pruning (in nm)
            tolerance_nm: Tolerance for simplification (in nm)
            num_workers: Number of parallel workers
            timeout: Timeout for ImageDataInterface reads
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
                     per-ID peak RSS, used to plan memory-aware dask waves.
                     Defaults fit on a c-elegans dataset (16nm isotropic
                     segmentation, ~16 B/voxel amplification not observed —
                     actual ~6.63 B/iso-voxel with a 250 MB library baseline).
                     For very different segmentation types or anisotropies,
                     re-profile and override.
            memory_fraction: Fraction of per-slot memory considered usable
                     when planning waves (rest is dask/OS/library overhead).
        """
        super().__init__(num_workers)
        self.segmentation_idi = ImageDataInterface(segmentation_path, timeout=timeout)
        self.output_path = str(output_path).rstrip("/")
        self.csv_path = csv_path
        # Normalize erosion parameter
        if erosion is True:
            erosion = "full"
        elif erosion is False or erosion is None:
            erosion = None
        elif erosion in (6, 18):
            erosion = str(erosion)
        elif erosion in ("full", "6", "18"):
            pass
        else:
            raise ValueError(
                f"erosion must be True, False, None, 'full', 6, 18, '6', or '18', got {erosion!r}"
            )
        self.erosion = erosion
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
        self.memory_fraction = float(memory_fraction)

        # Load CSV with bounding box info
        self.bbox_df = pd.read_csv(csv_path, index_col=0)
        self.ids = self.bbox_df.index.tolist()

        # Create output directories
        # Each output directory (full and simplified) needs its own structure
        os.makedirs(f"{output_path}/full", exist_ok=True)
        os.makedirs(f"{output_path}/full/segment_properties", exist_ok=True)
        os.makedirs(f"{output_path}/simplified", exist_ok=True)
        os.makedirs(f"{output_path}/simplified/segment_properties", exist_ok=True)

        logger.info(f"Loaded {len(self.ids)} IDs from {csv_path}")
        logger.info(f"Output will be written to {output_path}")

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
        erosion: str,
        min_branch_length_nm: float,
        tolerance_nm: float,
        sharded: bool = False,
    ):
        """
        Process a single ID: extract, skeletonize, prune, simplify, and emit.

        When ``sharded=True``, encoded skeleton bytes are returned in the
        result dict under ``"full_bytes"``/``"simplified_bytes"`` so the
        driver can pack them into shard files via the existing pickle merge
        path — no per-ID NRS write happens. When ``sharded=False``, per-ID
        files are written under ``{output_path}/{full,simplified}/{id}``.
        """
        from funlib.geometry import Roi

        result: dict = dict(Skeletonize._empty_metrics())

        def emit(subdir: str, skel_obj: CustomSkeleton):
            encoded = skel_obj.encode_neuroglancer_bytes()
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
            emit("simplified", empty)

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
            # Bounding box coords from CSV are in true nm; convert to scaled coordinates
            voxel_size = segmentation_idi.voxel_size
            sf = segmentation_idi.voxel_size_scale_factor
            padding = voxel_size  # 1 voxel in each direction
            start_point = np.array([min_z * sf, min_y * sf, min_x * sf]) - padding
            end_point = np.array([max_z * sf, max_y * sf, max_x * sf]) + padding
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

            # Compute EDT on pre-erosion mask for approximate radii
            distance_transform = edt_module.edt(data, anisotropy=tuple(isotropic_voxel_size))

            # Apply erosion if requested
            if erosion == "full":
                # Define a 3D cross-shaped structuring element (6-connectivity)
                cross_3d = np.array(
                    [
                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    ],
                    dtype=bool,
                )
                data = binary_erosion(data, cross_3d)
            elif erosion == "6":
                data = remove_unbridged_adjacencies(data, connectivity=6)
            elif erosion == "18":
                data = remove_unbridged_adjacencies(data, connectivity=18)

            if erosion is not None and not np.any(data):
                logger.warning(
                    f"Erosion removed all voxels for ID {id_value}, emitting empty skeleton"
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
                    emit("full", seed_skel)
                    emit("simplified", seed_skel)
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

            result["longest_shortest_path_nm"] = longest_shortest_path
            result["num_branches"] = num_branches
            result["radius_mean_nm"] = float(np.mean(radii))
            result["radius_std_nm"] = float(np.std(radii))

            # Simplify
            if tolerance_nm > 0:
                simplified = pruned.simplify(tolerance_nm)
            else:
                simplified = pruned

            if len(simplified.vertices) == 0:
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

            if len(simplified.edges) == 0:
                simplified.edges = np.zeros((0, 2), dtype=np.uint32)
            else:
                simplified.edges = np.array(simplified.edges, dtype=np.uint32)

            emit("full", skeleton)
            emit("simplified", simplified)
            return result

        except Exception as e:
            logger.error(f"Error processing ID {id_value}: {e}", exc_info=True)
            raise

    def write_neuroglancer_info_files(self):
        """
        Write the neuroglancer info file and segment_properties info file for both full and simplified directories.
        """
        # Write info files for both 'full' and 'simplified' directories
        for subdir in ["full", "simplified"]:
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

            if self.sharded:
                from cellmap_analyze.util.sharded_skeleton import make_sharding_spec
                info["sharding"] = make_sharding_spec(
                    shard_bits=self.shard_bits, minishard_bits=self.minishard_bits
                )

            info_path = f"{self.output_path}/{subdir}/info"
            with open(info_path, "w") as f:
                json.dump(info, f)
            logger.info(f"Wrote neuroglancer info file to {info_path}")

            # Write segment_properties info file
            segment_ids = [str(id_val) for id_val in self.ids]
            segment_properties_info = {
                "@type": "neuroglancer_segment_properties",
                "inline": {
                    "ids": segment_ids,
                    "properties": [
                        {
                            "id": "label",
                            "type": "label",
                            "values": ["" for _ in segment_ids],
                        }
                    ],
                },
            }

            segment_properties_path = (
                f"{self.output_path}/{subdir}/segment_properties/info"
            )
            with open(segment_properties_path, "w") as f:
                json.dump(segment_properties_info, f)
            logger.info(
                f"Wrote segment_properties info file to {segment_properties_path}"
            )

    def _estimate_peak_bytes(self, id_value):
        """Estimate per-ID peak RSS from the cached bbox row.

        Uses the isotropic voxel count of the (padded) bbox, which is what
        ``_skeletonize_id`` actually operates on after the optional zoom
        step. For isotropic datasets this equals the native voxel count.
        """
        row = self.bbox_df.loc[id_value]
        vs_nm = self.segmentation_idi.voxel_size
        original_vs = self.segmentation_idi.original_voxel_size
        # +2 voxels of padding matches _skeletonize_id's 1-voxel pad each side.
        dx = (row["MAX X (nm)"] - row["MIN X (nm)"]) + 2 * vs_nm[0]
        dy = (row["MAX Y (nm)"] - row["MIN Y (nm)"]) + 2 * vs_nm[1]
        dz = (row["MAX Z (nm)"] - row["MIN Z (nm)"]) + 2 * vs_nm[2]
        # If anisotropic, _skeletonize_id resamples to min(original_vs)
        # before EDT/skeletonize. That's the working voxel grid.
        min_vs = min(original_vs)
        iso_voxels = (dx * dy * dz) / (min_vs ** 3)
        return int(self.peak_bytes_baseline + self.peak_bytes_per_voxel * iso_voxels)

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

        tmp_merge_root = f"{self.output_path}/_tmp_skeleton_metrics_to_merge"
        all_metrics = []

        for wave_index, wave in enumerate(waves, start=1):
            phase_label = (
                f"skeletonize wave {wave_index}/{len(waves)} "
                f"({len(wave.item_ids)} IDs, procs/slot={wave.processes})"
            )
            wave_ids = wave.item_ids
            wave_merge_dir = f"{tmp_merge_root}_wave{wave_index}"

            def _wrapper(idx, _wave_ids=wave_ids):
                return self._skeletonize_id_by_value(_wave_ids[idx])

            def _phase(workers, config, _wrapper=_wrapper, _ids=wave_ids,
                       _merge=wave_merge_dir, _label=phase_label):
                return dask_util.compute_blockwise_partitions(
                    len(_ids), workers, self.compute_args, logger, _label,
                    _wrapper,
                    merge_info=(Skeletonize._merge_skeleton_metrics, _merge),
                    config=config,
                )

            wave_metrics = dask_util.run_with_oom_retry(
                _phase, wave.workers, phase_label, logger,
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

    def _skeletonize_id_by_value(self, id_value):
        """Dispatch one ID to ``calculate_id_skeleton``. Each wave dispatches
        a subset of IDs, so the wrapper takes the ID value directly rather
        than an index into ``self.ids``."""
        result = Skeletonize.calculate_id_skeleton(
            id_value,
            self.segmentation_idi,
            self.bbox_df,
            self.output_path,
            self.erosion,
            self.min_branch_length_nm,
            self.tolerance_nm,
            sharded=self.sharded,
        )
        if result is None:
            result = Skeletonize._empty_metrics()
        result["id"] = id_value
        return result

    def _pack_shards_from_metrics(self, metrics_list):
        """Pack encoded skeleton bytes (already in memory via pickle merge)
        into precomputed sharded shard files.

        Pops ``full_bytes``/``simplified_bytes`` from each metric dict in
        ``metrics_list`` so subsequent CSV writing sees only metric columns.
        No NRS read/unlink work — the bytes were carried back on the dask
        merge path that runs for every job regardless.
        """
        import time
        from cellmap_analyze.util.sharded_skeleton import pack_sharded_skeletons

        for subdir, bytes_key in [("full", "full_bytes"), ("simplified", "simplified_bytes")]:
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


# Splitting accidentally-merged objects at narrow bridges — design plan

Status: design discussion, not yet implemented. Written up so the next session
doesn't have to re-derive the reasoning.

## Goal

Some segmented objects (nuclei, cells, mitochondria) end up incorrectly fused
where two separate instances touch through a thin neck. We want a post-processing
step that:

- Preserves all original foreground voxels — only instance labels change.
- Splits true bridges (two large masses joined by a thin connection).
- Avoids over-splitting from boundary noise, wobble, or small spurs.

## Two object regimes need two strategies

A single fixed-radius "thin neck" rule doesn't work uniformly:

- **Nuclei / cells**: blob-like, not normally branched. A local narrowing is
  almost always a merge artifact. A classic **distance-transform watershed**
  (EDT + seeded watershed on local maxima) is the standard, simplest reliable
  approach here.
- **Mitochondria**: thin and branched everywhere, so raw local thickness can't
  distinguish "normal tubule" from "merge artifact" — a segment can be thin
  and totally legitimate. This needs a **skeleton/graph** approach that reasons
  about topology (does cutting this edge separate two branch-heavy, high-mass
  subtrees, or just prune a small twig?), not just a per-voxel radius check.

Both strategies share the same outer scaffolding (read object → find candidate
subpieces → gate by size → relabel) and only differ in how candidate cuts are
identified. Plan is to implement them as pluggable strategies behind one driver
class rather than as two disconnected tools.

## Open design questions (unresolved — flag before implementing)

We are **not confident** that hard, globally-fixed thresholds are the right
knobs, and want to leave room to change this without a rearchitecture:

- **Neck/thinness threshold**: still likely needed in some form (both
  strategies need *some* notion of "this is anomalously thin"), but a single
  fixed `neck_radius_nm` across all objects of a class may be wrong — a small
  nucleus and a huge nucleus don't have the same natural scale. Worth
  experimenting with **adaptive/relative thresholds** (e.g. relative to the
  object's own radius distribution, a percentile cutoff, or automatic
  bimodality detection in the per-object EDT/radius histogram) instead of one
  global constant. "Decide on the fly" per-object rather than one number for
  everything.
- **`minimum_subregion_volume`**: should be an **optional** gate (`None` /
  disabled by default or configurable per run), not a mandatory parameter —
  we're not sure yet whether it's the best way to reject spurious splits, or
  whether it should be expressed as voxel/nm³ volume (matches the rest of the
  repo's convention) vs. something else (e.g. skeleton path length for the
  branched mitochondria case, since "volume" is a less natural size metric for
  thin tubular subregions).
- General posture: treat the first implementation as a baseline to experiment
  against on real data, not a final answer — keep the threshold/gating logic
  isolated (see strategy interface below) so swapping in an adaptive version
  later doesn't require touching the driver, I/O, or relabeling machinery.

## Proposed architecture

Grounded in this repo's existing conventions (surveyed from
`connected_components.py`, `mutex_watershed.py`, `skeletonize.py`,
`dask_util.py`, `mixins.py`, `image_data_interface.py`,
`clean_connected_components.py`).

### Module: `src/cellmap_analyze/process/split_narrow_bridges.py`

#### `SplitNarrowBridges(ComputeConfigMixin)` — shared driver

Per-**object** parallelism (like `Skeletonize`), not spatial blocks (like
`ConnectedComponents`) — splitting needs the whole connected component in
memory at once.

```python
class SplitNarrowBridges(ComputeConfigMixin):
    def __init__(self, segmentation_path, output_path,
                 strategy,                              # "edt_watershed" | "skeleton_graph"
                 neck_radius_nm,                         # candidate-cut threshold (see open questions)
                 minimum_subregion_volume_nm_3=None,     # optional accept-gate (see open questions)
                 mask_config=None, bbox_csv_path=None,
                 num_workers=10, delete_tmp=False):
        ...
```

- Constructor mirrors `ConnectedComponents`: wrap `segmentation_path` in
  `ImageDataInterface`, convert nm params to voxels via `original_voxel_size`,
  resolve/generate `bbox_csv` the way `Skeletonize._generate_bbox_csv()` does,
  build a `strategy` instance from a small registry.
- **Output init**: copy `segmentation_path` → `output_path` wholesale up
  front (cheap blockwise copy) so untouched objects need zero writes.
- **Driver method** `split_objects()`: one dask task per object ID via
  `dask_util.plan_memory_waves` + `run_with_oom_retry` (same pattern as
  `Skeletonize`, for the same reason — per-object memory cost varies a lot).
- **Per-object static worker** `split_id(id_value, bbox_row, segmentation_idi,
  output_idi, strategy, neck_radius_voxels, min_subregion_voxels)`:
  1. Read padded bbox ROI, `mask = data == id_value`.
  2. `subpieces = strategy.find_subpieces(mask, voxel_size, neck_radius_voxels,
     min_subregion_voxels)` → 0/1..k label array, same shape as mask.
  3. `k <= 1`: no-op (output already has the original label from the copy step).
  4. `k >= 2`: assign `new_id = id_value * MAX_SUBPIECES + subpiece_index`
     (same global-offset trick used in `ConnectedComponents`/`MutexWatershed`
     for block IDs), read-modify-write only voxels where `mask` is true within
     the bbox (avoids clobbering neighboring objects sharing the same padded
     box), write to `output_idi`.
- **Final compaction**: new IDs never collide (deterministic offset per parent
  ID), so no union-find merge pass is needed. Optionally run `fastremap.renumber`
  for compact IDs, or reuse `ConnectedComponents.relabel_dataset` machinery
  directly rather than reimplementing.

#### Strategy interface (pluggable, single method)

```python
class SplitStrategy(ABC):
    def find_subpieces(self, mask, voxel_size, neck_radius_voxels,
                        min_subregion_voxels) -> np.ndarray: ...
```

**`EDTWatershedSplit`** (nuclei / cells):
- `edt.edt(mask, anisotropy=voxel_size)` (same call already used in
  `Skeletonize.calculate_id_skeleton`).
- `peak_local_max` on the EDT, `min_distance` derived from `neck_radius_voxels`,
  markers via `ndimage.label`.
- `skimage.segmentation.watershed(-distance, markers, mask=mask)`.
- Shared helper `merge_small_fragments(labels, min_subregion_voxels)`:
  iteratively dissolves any fragment below the volume gate into its
  largest-bordering neighbor (via `fastremap`/`ndimage`) until stable — this
  *is* the "only accept a cut if both sides are large" gate, applied
  post-hoc instead of during neck detection. No-op if the gate is disabled.

**`SkeletonGraphSplit`** (mitochondria):
- Reuses the shared prefix of `Skeletonize.calculate_id_skeleton` (isotropic
  resample, `edt`, `skimage.skeletonize`, `skimage_to_custom_skeleton_fast`)
  through `skeleton.skeleton_to_graph()` — factor that prefix into a shared
  helper used by both `Skeletonize` and this strategy, rather than duplicating it.
- New logic on `CustomSkeleton`: find candidate cut-edges where node radius
  `< neck_radius_voxels`; for each, check whether removing it splits the graph
  into ≥2 components that each clear the size gate (reuse
  `find_branchpoints_and_endpoints` for topology + a per-subgraph size
  estimate); accept/reject accordingly.
- Voxel assignment: nearest-skeleton-node labeling (KDTree over surviving
  subgraph node positions) turns the pruned graph back into a per-voxel label
  array — this is the "propagate labels back through the mask" step, done via
  nearest-neighbor rather than a bespoke flood-fill.

### Reuse map (avoid reimplementing)

| Need | Existing utility |
|---|---|
| EDT | `edt.edt(mask, anisotropy=voxel_size)` — used inline in `skeletonize.py:634-636`, no wrapper exists yet |
| Global unique IDs across independent units | block-index offset trick in `connected_components.py`/`mutex_watershed.py` → same idea, keyed by parent object ID |
| Per-ID voxel counts | `ConnectedComponents.get_object_sizes` |
| Volume-based accept/reject | `ConnectedComponents.volume_filter_connected_ids` |
| Relabeling at scale | `ConnectedComponents.write_memmap_relabeling_dicts` / `get_updated_relabeling_dict` / `relabel_dataset` |
| Per-object dask dispatch w/ memory variance | `dask_util.plan_memory_waves` + `run_with_oom_retry` (used by `Skeletonize`) |
| Skeleton graph / branchpoints / radii | `CustomSkeleton` in `util/skeleton_util.py` |

### CLI wiring

Follows the exact template every other step uses:
- `cli.cli.split_narrow_bridges()` function using `RunProperties` (config dir →
  `run-config.yaml` + `dask-config.yaml`, `ClassName(**run_config)`).
- New `[project.scripts]` entry in `pyproject.toml`.
- Config picks `strategy: edt_watershed` for nuc/cell runs and
  `strategy: skeleton_graph` for mito, via the same `run-config.yaml`
  convention as every other process class.

## Next steps

1. Decide (or defer/experiment with) the threshold questions above.
2. Implement `EDTWatershedSplit` first (simpler, more standard, unblocks
   nuc/cell testing).
3. Implement `SkeletonGraphSplit`, factoring the shared skeleton-building
   prefix out of `Skeletonize` rather than duplicating it.
4. Wire up CLI + pyproject entry point.
5. Test on real merged nuc/cell/mito instances before locking in defaults.

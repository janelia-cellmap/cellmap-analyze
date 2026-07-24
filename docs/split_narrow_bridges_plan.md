# Splitting accidentally-merged objects at narrow bridges — design plan

Status: implemented on branch `split-narrow-bridges`
(`src/cellmap_analyze/process/split_narrow_bridges.py`), both strategies
(`edt_watershed` and `skeleton_graph`) working and tested
(`tests/operations/test_split_narrow_bridges.py`). Written up so the next
session doesn't have to re-derive the reasoning.

## Goal

Some segmented objects (nuclei, cells, mitochondria) end up incorrectly fused
where two separate instances touch through a thin neck. We want a post-processing
step that:

- Preserves all original foreground voxels — only instance labels change.
- Splits true bridges (two large masses joined by a thin connection).
- Avoids over-splitting from boundary noise, wobble, or small spurs.

## Two strategies, not two organelle-specific tools

Neither strategy is specific to any organelle type — both are general-purpose
and both run on any object shape. The reason to have two is that a single
fixed-radius "thin neck" rule doesn't work well across very different
morphologies:

- **`edt_watershed`**: the classic distance-transform watershed (EDT + seeded
  watershed on local maxima). Simplest, most standard, and the natural
  default — works well whenever an object is blob-like (not normally
  branched), where a local narrowing is almost always a merge artifact.
  Nuclei and cells are the common case, but nothing about it assumes that.
- **`skeleton_graph`**: topology-aware — builds the object's skeleton graph
  and only treats a point as a candidate neck if it's a genuine **local
  minimum** of radius (thinner than every neighbor), not merely "thin." This
  matters most for thin/branched shapes (mitochondria networks are the
  motivating case) where raw local thickness can't distinguish "normal
  tubule" from "merge artifact" — but it works fine on simple blobs too (see
  "what we learned" below), so it's a reasonable strategy to reach for
  whenever topology, not just local radius, should decide the cut.

Both strategies share the same outer scaffolding (read object → find candidate
subpieces → gate by size → relabel) and only differ in how candidate cuts are
identified — implemented as pluggable strategies behind one driver class
rather than as two disconnected tools.

## What we learned while implementing `skeleton_graph` (important)

The first version compared each skeleton node's radius against a flat
`neck_radius` threshold and cut any qualifying edge. That's wrong: a solid
blob with a thin spur/protrusion has EDT radius shrinking *monotonically*
from the blob's core down to the spur's tip — thin nearly everywhere along
that taper, even though there's no second object on the far side (no
dip-then-rise, just a shape narrowing to a point). A bare threshold flagged
these constantly, which is exactly the over-splitting failure mode this
project set out to avoid.

The fix: a candidate neck must be a genuine **local minimum** — a node whose
radius is less than or equal to *every* neighbor's, with degree ≥ 2 (skeleton
endpoints are excluded outright — nothing continues past them, so there's no
"far side"). This correctly separates a real bridge (thick-thin-thick) from
an ordinary taper (monotonically thin-to-a-point). Cutting happens by
removing the neck *node* (not an edge) from the skeleton graph, since the
neck is conceptually a point, not a specific connection.

The size-gate accounting also went through two wrong attempts before landing
on an accurate one:
1. A `pi * r^2 * length` tube-volume estimate per skeletal edge — overestimated
   a real 3-voxel stub as ~6 "voxels" (radius-1 cross-sections aren't disks of
   area π), enough to slip past the gate.
2. Straight-line Euclidean nearest-skeleton-node assignment for every mask
   voxel — leaked most of a compact cube's own volume onto a thin stub's
   skeleton nodes, because straight-line distance ignores the mask's actual
   shape/connectivity.

What actually works: assign every mask voxel to its nearest skeleton node via
a **marker-based watershed on a flat field, restricted to the mask**
(`skimage.segmentation.watershed(np.zeros(mask.shape), markers, mask=mask)`)
— a geodesic, through-the-mask nearest-seed assignment. Summing the resulting
per-node voxel counts over a candidate component gives an exact voxel count,
not a geometric proxy.

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
- **Per-object static worker** `split_id(id_value, segmentation_idi, bbox_df,
  strategy, neck_radius_voxels, minimum_subregion_volume_voxels,
  max_pieces_per_object, scratch_dir)`:
  1. Read padded bbox ROI, `mask = data == id_value`.
  2. `subpieces = strategy.find_subpieces(mask, original_voxel_size,
     neck_radius_voxels, minimum_subregion_volume_voxels)` → 0/1..k label
     array, same shape as mask.
  3. `k <= 1`: no-op, return `{"split": False}`.
  4. `k >= 2`: persist the local `subpieces` array to a small per-object
     scratch `.npz` and return its path + bbox roi. **Actual writing is
     deferred** — see below for why.
- **Why two phases, not a direct write**: writing per-object results straight
  into a shared zarr array from concurrent per-object workers would race —
  two objects with overlapping padded bboxes can share a chunk, and a
  concurrent partial-chunk read-modify-write is a real data race (last
  writer wins, silently dropping the other's edit). Fixed by splitting into:
  **phase 1** (per-object, as above) computes into scratch files only;
  **phase 2** is a single blockwise pass, spatially disjoint per block (same
  tiling discipline as `ConnectedComponents`), applying all accepted splits
  by reading `segmentation_idi` (never the not-yet-written output) block by
  block and overlaying any split object's scratch labeling that intersects
  that block.
- **Final ID assignment**: rather than an `id_value`-derived offset (risk of
  collision with an untouched object's real ID, since IDs aren't necessarily
  dense/sequential), each split object gets a block of `max_pieces_per_object`
  new IDs starting strictly above `max(all_original_ids)` — computed
  driver-side, after all objects are scanned, so no union-find merge pass is
  needed and new IDs can never collide with an untouched object's ID.

#### Strategy interface (pluggable, single method)

```python
class SplitStrategy(ABC):
    def find_subpieces(self, mask, voxel_size, neck_radius_voxels,
                        min_subregion_voxels) -> np.ndarray: ...
```

**`EDTWatershedSplit`**:
- `edt.edt(mask, anisotropy=voxel_size)` (same call already used in
  `Skeletonize.calculate_id_skeleton`).
- `peak_local_max` on the EDT, `min_distance` derived from `neck_radius_voxels`,
  markers via manual seeding, `skimage.segmentation.watershed(-distance,
  markers, mask=mask)`.
- Helper `_merge_small_fragments(labels, min_volume_voxels)`: iteratively
  dissolves the smallest fragment below the gate into its largest-bordering
  neighbor (via `fastremap`/`scipy.ndimage.binary_dilation`) until every
  surviving fragment clears it — this *is* the "only accept a cut if both
  sides are large" gate, applied post-hoc instead of during neck detection.
  No-op if the gate is disabled (`None`).

**`SkeletonGraphSplit`**:
- `edt.edt` + `skimage.morphology.skeletonize` + `skimage_to_custom_skeleton_fast`
  + `CustomSkeleton.skeleton_to_graph()` to get a graph with per-node radius.
- Candidate necks are **nodes** (not edges) that are a genuine local minimum
  of radius among their neighbors (degree ≥ 2, radius < threshold, radius ≤
  every neighbor's) — see "what we learned" above for why a bare threshold
  doesn't work. Candidates are tried thinnest-first: temporarily remove the
  node, check `networkx.connected_components`, and (if the gate is set)
  require every resulting component's voxel count to clear it; otherwise
  restore the node and move on.
- Voxel accounting/assignment: one marker-based `watershed` call (flat field,
  restricted to the mask, markers at every skeleton voxel) assigns each mask
  voxel to its nearest node geodesically. Per-node voxel counts from this are
  summed per candidate component for the size gate, and the final label array
  reuses the same assignment (nodes cut out get resolved to whichever
  surviving neighbor's side they're topologically closest to).

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

## Status / next steps

Done: both strategies implemented and unit-tested on synthetic geometries
(dumbbells, a compact cube, a cube-with-spurious-stub); CLI (`split-narrow-bridges`)
and `pyproject.toml` entry wired up.

Not done yet:
1. Never run on real merged nuc/cell/mito data — only synthetic smoke tests
   so far. Needed before trusting any default threshold values.
2. `split_objects()` dispatches per-object work via a single
   `compute_blockwise_partitions` call, not `dask_util.plan_memory_waves` +
   `run_with_oom_retry` (the memory-aware wave scheduling `Skeletonize` uses
   for the same per-object-size-variance problem). Worth adding if large
   objects (e.g. big mitochondria networks) OOM in practice.
3. The threshold questions from "Open design questions" are still open —
   adaptive/relative thresholds haven't been explored; `neck_radius_nm` is
   still one fixed value per run.
4. `SkeletonGraphSplit`'s per-candidate-neck connectivity check is
   O(candidates × graph size) in the worst case (full `connected_components`
   recomputed per trial) — fine for per-object bbox-sized skeletons, but
   worth revisiting if very large branched networks turn out to be slow.

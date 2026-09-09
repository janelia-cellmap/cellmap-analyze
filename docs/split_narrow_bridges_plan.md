# Splitting accidentally-merged objects at narrow bridges — design plan

Status: implemented on branch `split-narrow-bridges`
(`src/cellmap_analyze/process/split_narrow_bridges.py`). Only `edt_watershed`
ships currently — `skeleton_graph` and `tube_direction` were removed before
merging (see "skeleton_graph and tube_direction removed" below); the design
history for them is kept in this doc since the reasoning (thin-segment
detection, cyclic-mesh handling, direction gates) is worth re-deriving from
if either comes back with real test/validation coverage. Written up so the
next session doesn't have to re-derive the reasoning.

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

## Boundary-thinness gate, real-data testing, and a 2.4hr perf bug

`EDTWatershedSplit` originally only gated splits by fragment *size*
(`_merge_small_fragments`); real cerebellum-nucleus data showed this isn't
enough — two peaks sitting in one continuously-fat, irregularly-shaped blob
(a lumpy real nucleus, not two merged objects) can each seed their own
watershed basin with no real neck between them. Added
`_merge_thick_boundaries`: for every pair of adjacent watershed regions,
find the widest point along their shared boundary and merge them back
together (worst offender first) until every remaining boundary is genuinely
thinner than `neck_radius_nm`. This is the "thin, between two thicker
things" check for `edt_watershed`, mirroring what `SkeletonGraphSplit`
already did topologically.

A first version of `_merge_thick_boundaries` recomputed every region's mask
+ dilation + neighbor boundaries from scratch after every single merge —
fine for a handful of fragments, but on a real ~100M-voxel merged-nucleus
object with ~50 initial peaks it took **2.4 hours**. Fixed by separating the
one-time expensive array work (a single pass over the array to find every
touching label pair and the max EDT radius at their shared boundary) from
the cheap iterative decision-making (union-find merging on that small
label-adjacency graph) — same real object now takes 216s, bit-identical
output (~40x speedup).

Tested on real `jrc_mus-cerebellum-2` nucleus data (ids 2307, 2131).
`skeleton_graph` reliably under-splits relative to `edt_watershed` on lumpy
blob shapes: Lee's 3D thinning creates spurious topological loops/cycles on
bumpy real biological surfaces, letting genuine thin necks be "bypassed"
(cutting the candidate segment doesn't disconnect the graph, since the loop
provides another path around it) even when correctly identified as a
candidate. This is a deeper limitation than neck-detection logic and isn't
fixed by better thresholds — `skeleton_graph` remains most reliable for
genuinely tree-like/branched shapes (its motivating mitochondria case), not
lumpy blobs (nuclei/cells), where `edt_watershed` is the safer default.

## `ComputeEDT` and wiring precomputed EDT into `SplitNarrowBridges`

`edt.edt()` (the Seung-lab package) is multi-label aware: given the raw
labeled array directly — not a binarized single-instance mask — it computes,
per voxel, the distance to the nearest voxel with a *different* label
(background included). That's exactly what each strategy needs per object,
just computable once for the whole dataset instead of recomputed per object.
Added `src/cellmap_analyze/process/compute_edt.py`
(`ComputeEDT(ComputeConfigMixin)`, CLI `compute-edt`): a blockwise pass that
persists this multi-label EDT as its own dataset. Windowed/truncated EDT can
only *overestimate* the true distance (a min over a smaller candidate set),
so a fixed `padding_nm` only needs to comfortably exceed whatever thinness
threshold downstream consumers care about, not the physical scale of any
object — validated on real id 2307 data (100% agreement with the true
per-object EDT on the actual "<500nm" threshold decision that
`SplitNarrowBridges` makes, despite raw value differences up to 253.5nm deep
in thick interiors).

`SplitNarrowBridges` now takes an optional `edt_path`: when given, each
object reads its EDT window from there (`ImageDataInterface.to_ndarray_ts`
over the same roi used to read the segmentation) instead of calling
`edt.edt(mask, ...)` itself — the expensive part for very large merged
objects (e.g. the 97M-voxel id 2131). `SplitStrategy.find_subpieces` grew an
optional `distance=None` parameter threaded through both strategies; `None`
falls back to the original per-object computation.

When `edt_path` is *not* given, `precompute_edt` (**default `False`** — see
below for why this flipped from the original `True`) controls whether
`SplitNarrowBridges` auto-runs `ComputeEDT` once, up front, into a scratch
dataset (named/cleaned up the same way the other scratch files/dirs already
are, via `_run_id`/`delete_tmp`) instead of the bare per-object recompute —
the same "auto-run the expensive prerequisite if not supplied" pattern
already used for `csv_path=None` auto-running `Measure`. `edt_padding_nm`
(default `max(neck_radius_nm, one voxel)`) controls the padding for that
auto-run. Set `precompute_edt=True` if you're iterating on thresholds
across many runs over the same segmentation and want to amortize the EDT
cost (better: run `compute-edt` yourself once and pass the result via
`edt_path`, reused every run). A per-object shape mismatch between the
precomputed dataset and the segmentation crop raises rather than silently
misaligning (should never happen if `edt_path` was produced by `ComputeEDT`
run over the same `segmentation_path`).

**`precompute_edt` default flipped `True` → `False` after a real production
failure.** A real run on `jrc_mus-cerebellum-3`'s `nuc` dataset (32nm voxels,
256³ chunks, `neck_radius_nm=1000` → `edt_padding_nm=1000`) OOM'd during the
`precompute_edt=True` `ComputeEDT` blockwise pass: `block_multiplier=4` gives
1024³-voxel native blocks, plus 32 voxels of padding on every axis/side (from
`ceil(1000nm / 32nm)`) → a padded read of 1088³ ≈ 1.29B voxels per block —
against a 240GB/16-process ≈ 14GB per-slot budget in `dask-config.yaml`.
Unlike `SplitNarrowBridges`'s own per-object dispatch (`_estimate_peak_bytes`
+ `plan_memory_waves`), `ComputeEDT.calculate_edt()` has **no memory-aware
wave planning at all** — every block gets the same fixed per-slot budget
regardless of how big `block_multiplier`/`padding_nm` make it, and
`calculate_block_edt` (`compute_edt.py:110`) never `del`s the raw `data`
array and does a redundant `.astype(np.float32)` copy even though
`edt.edt()` already returns float32 for both boolean-mask and multi-label
integer input (confirmed empirically — not float64, as this doc's own
per-object estimator assumed elsewhere). Combined, the per-block peak
(~12GB) left almost no margin, and cross-block allocator fragmentation
within the same long-lived worker process (see the "Bytes stored"/unmanaged-
memory investigation below) was enough to tip it over — 4 different workers
died on the same block before dask gave up.

Root cause is structural, not a tunable-away fluke: `ComputeEDT` dispatches
on a *fixed grid*, so every block pays the same `neck_radius_nm`-scaled
padding regardless of whether it's actually near a real object boundary —
deep-interior blocks of a huge merged blob cost exactly as much padding as
blocks that are genuinely thin. Per-object dispatch doesn't have this
problem: `_object_roi`'s crop is tight around each object's *own* real
extent, so equivalent padding is comparatively cheap, and it already flows
through the proven-safe per-object wave machinery for free. Rather than
build memory-aware wave planning for `ComputeEDT` too (feasible — since
every block is ~uniform size here, unlike per-object bbox variance, it would
reduce to a single upfront sizing decision, not true multi-wave scheduling —
but still new code, new surface area), the fix was simply to default to the
path that's already safe: `precompute_edt=False`, per-object. `ComputeEDT`
remains available (`precompute_edt=True`, or run standalone + `edt_path=`)
for the genuine amortization use case, with its memory-safety caveat now
documented above and in its own docstring.

**Bug found while validating this wiring**: `_object_roi`'s "1 voxel
padding in each direction" was silently asymmetric. Measure's
`"MIN/MAX (nm)"` CSV columns are voxel-*center* coordinates, not edges. The
original code did `MIN*sf - padding` / `MAX*sf + padding`: since `MIN*sf` is
already half a voxel *inside* the object's true bounding box, subtracting a
full voxel of padding correctly lands a full voxel past the low edge — but
`MAX*sf` is likewise half a voxel *inside* the box, so adding a full voxel
only reached the far edge of the object's own last voxel, giving **~zero**
real margin on the high side of every axis. This meant the per-object mask
EDT (`edt.edt(mask, ...)` on the tightly-cropped bbox) silently lost
accuracy for voxels near the high-coordinate faces of every object — an
edge-truncation artifact, invisible until compared against `ComputeEDT`'s
globally-correct output (max diff up to 5 voxels' worth of distance on a
20³ synthetic test). Fixed in `_object_roi` by shifting both bounds by
`voxel_size/2` before applying padding, landing them exactly on grid edges
regardless of rounding downstream.

The identical high-side bug existed in `Skeletonize.calculate_id_skeleton`
(`skeletonize.py:573-588`, same variable names, same `- padding`/`+ padding`
structure) and was fixed there too -- but *not* with the same symmetric
shift-both-bounds approach. Skeletonize's final vertex coordinates are
computed as `index * spacing + start_point_nm` (`skeletonize.py:~717-728`),
where `index * spacing` is an edge-relative local coordinate from
`skimage_to_custom_skeleton_fast`. It turns out `start_point`'s *un-shifted*
form (`MIN*sf - padding`, sitting exactly `voxel_size/2` below the true low
edge) is not a bug for that call site — it's exactly the correction needed
to convert the edge-relative local coordinate into an absolute voxel
*center* coordinate, and an existing regression test
(`test_skeletonize_nonzero_translation_shifts_by_translation`) encodes this
as a correctness requirement. Symmetrically "fixing" the low side there
broke that test (vertices landed half a voxel low). So the Skeletonize fix
only touches `end_point` (`MAX*sf + voxel_size/2 + padding`) — that side only
controls how much extra background the read grabs, not any coordinate
offset used later, so it was safe to correct without touching `start_point`.

## Skipping ginormous false-positive objects (removed)

Originally added a `max_object_volume_nm_3` pre-filter to skip objects above
a volume threshold entirely, before any per-object work was dispatched —
intended for segmentation noise/artifacts connected into one enormous blob,
not a real merged nucleus/cell/mito.

**Removed.** In practice, "this ConnectedComponents object is too large"
usually means it's several *real* objects legitimately merged via thin
bridges — exactly what this tool exists to split. Pre-filtering by size
before splitting throws out precisely the objects most in need of it,
before they get a chance. The right place for a volume cap is *after*
splitting, not before: `minimum_volume_nm_3`/`maximum_volume_nm_3` (deferred
`CleanConnectedComponents` pass) let an oversized merged blob split into
properly-sized real pieces first, and only drop what's *still* oversized
afterward — at which point it really is a strong signal of genuine noise,
not a legitimate object discarded prematurely. For a standalone size-based
cull with no splitting involved at all, use `filter_ids`/
`CleanConnectedComponents` directly on the `ConnectedComponents` output.

## Skipping tiny objects before splitting (added)

Added `minimum_object_volume_to_split_nm_3`: an optional pre-filter checked
against each object's *total* volume (from the bbox CSV, before any mask is
even read) — objects below it are never handed to a strategy at all, and are
written through unchanged.

This is the mirror image of the removed `max_object_volume_nm_3` idea above,
but not subject to the same objection. Skipping *large* objects up front was
wrong because "too large" is usually the exact signature of several real
objects merged together — the thing this tool exists to fix. Skipping
*small* objects up front doesn't have that problem: an object too small to
contain two real merged things almost never does, and `minimum_subregion_volume_nm_3`
already forces this mathematically anyway (a split needs >= 2 pieces, each
clearing that gate, so the object needed >= ~2x that volume to survive the
existing per-cut gate in the first place). The pre-filter doesn't change
*which* objects end up split — it just skips the mask/EDT/skeleton work for
objects that were always going to fail the existing gate after that work was
done, and stays independently useful as its own knob if
`minimum_subregion_volume_nm_3` is explicitly disabled (`0`).

## skeleton_graph and tube_direction removed (insufficiently tested)

Both `SkeletonGraphSplit` and `TubeDirectionSplit` (which subclassed it,
though it overrode every method and didn't actually rely on inherited
behavior) were removed before merging this branch.

- `SkeletonGraphSplit` had only two unit tests (a basic dumbbell split and
  one size-gate rejection on synthetic geometry) — no test of the cyclic
  topology its own docstring says it handles incorrectly (one-at-a-time
  cutting can't separate two blobs joined by *two* redundant thin bridges;
  see `TubeDirectionSplit`'s docstring for the real-data case that motivated
  its simultaneous-cut design), and no real-dataset validation run recorded
  anywhere.
- `TubeDirectionSplit` had **zero** unit tests. Its docstring cites specific
  numbers from a real object (id 1278, `jrc_axolotl-heart-1` mito, 6.4um
  crop — 380 candidate bridges, 35 vs. 157 disconnected one-at-a-time vs.
  simultaneously, piece counts stable at 82/81/81/71 across a
  `neck_radius_nm` range) that read like a genuine analysis was run during
  development, but it isn't captured as a reproducible test, so those claims
  can't be re-verified now and nothing guards against regressing them.

Rather than ship two strategies whose only evidence of correctness is
synthetic dumbbell tests (`skeleton_graph`) or docstring narrative with no
test at all (`tube_direction`), both were cut back out. `edt_watershed` is
the one strategy with real-data validation on record (see
`jrc_mus-cerebellum-2`/`-3` nucleus data below) and is what ships. If
thin/branched-object splitting (mitochondria, etc.) is needed again, the
design reasoning above (thin-segment/flanking detection, prominence and
direction gates, simultaneous cutting for cyclic meshes) is still valid —
it just needs unit tests covering the cyclic-topology case and a real
validation run captured somewhere reproducible before it ships again.

## Status / next steps

Done: both strategies implemented and unit-tested on synthetic geometries
(dumbbells, a compact cube, a cube-with-spurious-stub, an oversized-object
skip, precomputed-EDT wiring); CLI (`split-narrow-bridges`, `compute-edt`)
and `pyproject.toml` entries wired up; tested against real
`jrc_mus-cerebellum-2` nucleus data.

Also done:
1. Adaptive/relative neck threshold: opt-in `neck_radius_mode="adaptive"`
   Otsu-thresholds each object's own EDT values (floored at the fixed
   `neck_radius_nm`), so a small and a huge object don't have to share one
   scale. Default remains `"fixed"` (unchanged behavior). Experimental
   baseline, not a final answer — see `_adaptive_neck_radius_nm`.
2. `split_objects()` now dispatches per-object work through
   `dask_util.plan_memory_waves` + `run_with_oom_retry`, mirroring
   `Skeletonize`'s memory-aware wave scheduling, via a new
   `_estimate_peak_bytes` estimator.
4. Correct pipeline ordering for volume filtering: new
   `minimum_volume_nm_3`/`maximum_volume_nm_3` params run splitting with
   filtering deferred, then apply `CleanConnectedComponents` once over the
   fully-formed result — the same deferred-filtering pattern
   `MutexWatershed` uses with `do_opening`.
6. `minimum_subregion_volume_nm_3` now defaults (`None`) to
   `minimum_volume_nm_3` instead of disabling the accept-gate entirely: a
   split producing a piece below the final volume filter would just get
   *deleted* (voxels dropped, not re-merged) by the deferred
   `CleanConnectedComponents` pass anyway, so reject that split up front
   instead and keep those voxels attached to the parent object. When
   `minimum_volume_nm_3` is also left at its own default (0), this derives
   to 0 (falsy → gate still disabled), so the pure-defaults case is
   unchanged. Pass `minimum_subregion_volume_nm_3=0` explicitly to disable
   the gate while still using a real `minimum_volume_nm_3`.
7. `minimum_object_volume_to_split_nm_3` pre-filter (see above) — skips the
   split attempt (mask read, EDT, skeleton) entirely for objects below a
   total-volume threshold. Default 0 disables it (unchanged behavior).

Not done / explicitly out of scope:
3. `SkeletonGraphSplit`'s per-candidate-neck connectivity check is
   O(candidates × graph size) in the worst case (full `connected_components`
   recomputed per trial) — fine for per-object bbox-sized skeletons.
   Dropped from scope: an incremental-connectivity swap is more error-prone
   to verify than the one-time flat swap `get_connected_ids` did, and it's
   only worth it if very large branched networks turn out to be slow.
5. Scaling to ginormous merged objects (see below) — deferred, not built.
   The "beads on a string" design below is superseded by a simpler
   decision: item #2's memory-aware wave scheduling already gives an
   oversized object (e.g. real id 4665, ~4B-voxel bbox) a solo worker with
   the *entire* per-job memory (`dask_util.plan_memory_waves` floors at
   `processes=1` rather than an artificially small per-slot share), so it
   should just run through the existing, already-validated
   `EDTWatershedSplit` path unchanged. Only worth building the bespoke
   out-of-core algorithm below if that's tried on real id 4665 and actually
   proves insufficient (OOMs even on the largest obtainable job memory).

   **Now fully validated against the real object -- memory is not the
   blocker.** Ran `split_id` directly against real id 4665 in
   `jrc_mus-cerebellum-2`'s `nuc` dataset (confirmed present and matching
   the doc's bbox/volume exactly) twice:
   - On a 93GB workstation: real RSS climbed to ~69GB mid-`edt.edt()`
     computation (before even reaching the markers/watershed stages) before
     the OS killed the process — consistent with the ~68GB raw estimate.
     Confirmed the object is too big for that machine, but not whether a
     real high-memory allocation would work.
   - On a real LSF allocation (`bsub -n 18` on this cluster's slot model =
     270GB, see `scripts/smoketest_split_id_4665.py`): **ran to completion**
     in ~22 minutes, peak RSS 108.9GB (comfortably inside the 270GB
     budget, and in line with the ~kernel bytes/voxel estimate) -- no OOM.
     `edt.edt()`, `peak_local_max`, `watershed`, and `_merge_thick_boundaries`
     all completed over the full ~4B-voxel padded bbox in one shot.

   So: given a real 150GB+-class allocation, the existing, unmodified
   whole-object `EDTWatershedSplit` path handles the largest known real
   object with no OOM. **"Beads on a string" is not needed on memory
   grounds** -- the memory-aware wave scheduling (item #2) giving an
   oversized object a solo worker with a large job's full memory was
   already sufficient; no bespoke out-of-core algorithm is required.

   **What it hit instead is a different, more informative wall:**
   `ValueError: Object 4665 split into 122 pieces, exceeding
   max_pieces_per_object=64`. A real chain of accidentally-merged nuclei
   ("beads on a string") would split into a handful of nucleus-sized
   pieces, not 122 -- this is a strong signal that id 4665 specifically
   is one of the *ginormous false-positive background* blobs flagged at
   the very start of this effort (segmentation noise merged into one
   enormous mass), not a real merged-nuclei chain. This validates the
   deferred-filtering design already in place (splitting first with no
   pre-filter, then applying `minimum_volume_nm_3`/`maximum_volume_nm_3` via
   `CleanConnectedComponents` after the fact, see "Skipping ginormous
   false-positive objects" above): a real run wouldn't hand-tune
   `max_pieces_per_object` upward for this one object, it would let it
   split into however many fragments and rely on the final volume filter to
   drop whatever's still spurious. Re-ran with `max_pieces_per_object=256`
   so the split actually completes and persists: same 122 pieces, viewed in
   neuroglancer alongside the original object (see
   `scripts/smoketest_split_id_4665.py`) -- confirmed by eye to be
   consistent with a noise/artifact blob, not a real merged-nuclei chain.
   Default `max_pieces_per_object` raised from 64 -> 1024 in
   `SplitNarrowBridges.__init__` so real data like this isn't hand-tuned
   per object; it also doubles as the fixed new-ID stride per split object,
   so it can only be raised, not disabled outright (`None`), without
   reworking ID assignment.

   Also surfaced an unrelated prerequisite (since fixed): `ImageDataInterface`
   defaulting to `concurrency_limit=1` made the initial read of this
   object's ~23,000-chunk bbox time out entirely (5.5min, 10 retries) before
   any memory pressure even began. `SplitNarrowBridges`/`Skeletonize` now
   auto-resolve `concurrency_limit` (`dask_util.resolve_concurrency_limit`):
   every CPU actually available to the process when running synchronously
   (`num_workers<=1`, no sibling processes to oversubscribe), or a safe `1`
   under real multi-worker wave dispatch, further rescaled per-wave inside
   each worker to its fair share of its *job's* real CPU affinity
   (`dask_util.rescale_idi_concurrency`) -- since a job's cpuset is shared by
   every sibling worker process inside it, not divided per-process.

## Scaling to ginormous merged objects: bounded local growth ("beads on a string")

### The concrete problem

Real `jrc_mus-cerebellum-2` nuc data has objects far beyond anything tested so
far. From the dataset's own bbox CSV, id 4665: volume 12,673.55 µm³, but its
bounding box is 61 × 49 × 44 µm — roughly **4 billion voxels** at this
dataset's 32nm voxel size. The object fills only ~9.6% of that bbox (not one
compact blob — something sprawled across 60µm). The current architecture
(`split_id` reads the object's full padded bbox, computes EDT and runs
watershed over the whole thing at once) cannot handle this: the float64 EDT
array alone would be ~32GB for this one object, on top of the raw read
(~8GB), mask, markers, and label arrays. This isn't a "slow" case, it's
likely an outright OOM on typical worker memory.

### Two scaling ideas already considered, and their real costs

- **Erode-at-threshold + blockwise `ConnectedComponents`**: threshold the
  (already blockwise) `ComputeEDT` output at `neck_radius_nm`, erode those
  voxels away, run the existing blockwise `ConnectedComponents` on what's
  left to get separated "cores," then reassign the eroded shell back via a
  geodesic nearest-core watershed. Fully blockwise/scalable, but rejected in
  discussion because thresholding-and-discarding voxels outright (rather
  than a graded, competitive decision) isn't how real watershed reasons
  about a neck, and the discarded/no-surviving-core edge case is ugly.
- **Block-tiled local watershed + cross-block region-merge**: this is the
  *established* pattern in this exact scientific ecosystem — `daisy`
  (funkelab/HHMI Janelia, same lab as this repo's existing `funlib.geometry`
  dependency) is a blockwise task scheduler built for precisely this, and
  real large-scale connectomics pipelines run blockwise watershed for an
  initial oversegmentation, then blockwise agglomeration (e.g. `waterz`) to
  merge fragments via a region-adjacency graph. This repo already
  reimplements daisy's core idea by hand (`dask_util`'s blockwise dispatch +
  halo/padding), so only the merge-graph step would be new — and we already
  have that logic (`_merge_thick_boundaries`'s single-pass-boundary-scan +
  union-find), just scoped to one in-memory array instead of across blocks.
  Real cost: block boundaries are *arbitrary* relative to the object's own
  geometry, so a seed near a block edge or a real ridge running along a
  seam still needs the cross-block reconciliation step to sort out — solved,
  but it's inherent overhead that has nothing to do with the actual object
  shape.

### The better idea: bounded local growth ("beads on a string")

The key insight that neither idea above uses: **we only care about
nucleus-sized (or similarly bounded) pieces**, and a real accidentally-merged
chain of nuclei looks like beads on a string — a sequence of thick, roughly
nucleus-sized lobes, each one fully enclosed by genuine thin necks on every
side that connects it to its neighbors in the chain. If that's the actual
shape (and it very plausibly is for id 4665, given the low fill fraction and
elongated bbox), then **growing outward from a single seed and stopping the
instant the growing region is fully enclosed by neck-walls needs to touch
only that one bead's own local extent** — not the other 99% of a
4-billion-voxel object 60µm away that has nothing to do with it. The object
never needs to be loaded or reasoned about as a whole; each bead is
discovered, grown, sealed off, and finished independently, one at a time,
using memory proportional to *one bead's size* (nucleus-scale — thousands to
tens of thousands of voxels) rather than the whole merged mass.

Sketch of the algorithm:

1. **Cheap seed-finding, no full-object read**: scan the already-persisted,
   already-blockwise `ComputeEDT` output for local maxima belonging to this
   object's id — this can be done blockwise/lazily (same padding-can-only-
   overestimate guarantee `ComputeEDT` already relies on), without ever
   materializing the full object.
2. **Grow one bead from a single unclaimed seed**: a priority-flood (the
   same priority-queue-by-distance-from-seed structure `skimage.watershed`
   already uses internally), but reading the underlying segmentation/EDT
   lazily, chunk by chunk, only for whatever the growing frontier currently
   touches. A frontier voxel with EDT below `neck_radius_nm` is a wall —
   growth doesn't cross it (this bead's flood stops there), the same "thin
   between two thicker things" semantics `_merge_thick_boundaries`/
   `SkeletonGraphSplit` already use, just applied as a stopping rule during
   growth instead of an after-the-fact merge-back.
3. **Detect enclosure, declare the bead done**: once every direction the
   frontier could still expand into is either background or a neck-wall
   (no further unclaimed, above-threshold voxels reachable), this bead is
   fully bounded — finalize its label and stop. Nothing beyond this bead's
   own local footprint was ever touched.
4. **Move to the next bead**: pick the next unclaimed voxel of this same
   object id (again via the blockwise EDT/segmentation scan, not a
   full-object load) and repeat from step 2, walking along the "string" one
   bead at a time until every voxel of the original object has been
   claimed.

### Why this beats block-tiling

No arbitrary seams. A block-tiled approach still needs a merge step because
block boundaries have nothing to do with the object's real geometry — a
real basin can straddle an arbitrary tile edge. Here there's no tiling at
all: the "block" *is* the bead, sized exactly to its own true extent by the
enclosure-detection rule, so there's nothing arbitrary left to reconcile.

### The honest caveat (the user's own "potentially we'd need the whole
thing, but likely not")

This only pays off if the object actually *is* bead-like — bounded,
nucleus-sized lobes separated by genuine necks. If it's instead one
enormous, genuinely continuous mass with no internal neck structure (a true
segmentation blob artifact, or a single pathologically large real nucleus),
enclosure never triggers and growth doesn't stop until it has covered the
whole object anyway — at which point this degrades to exactly the current
whole-object approach, no worse than today. That's a good property: the
technique is a strict improvement in the expected/common case (real chains
of touching nuclei) with a fallback that never underperforms the baseline.

### Open questions before implementing

- **Next-seed discovery without a full scan**: finding "the next unclaimed
  voxel of this id" cheaply, incrementally, without ever doing an O(full
  bbox) pass, needs a real data structure (e.g. tracking claimed regions
  against the object's already-known bbox/chunk list from `ComputeEDT`'s
  own blockwise pass, so "not yet visited chunks" is always cheap to name).
- **Hard neck-wall threshold vs. the existing segment/boundary-thinness
  logic**: growth needs a *local* stopping rule cheap enough to check per
  frontier voxel, but the existing strategies' more careful "thin between
  two thicker things" checks operate on a whole candidate region/segment
  after the fact. Need to work out whether a simple per-voxel EDT threshold
  during growth is good enough, or whether some bounded lookahead is needed
  to avoid a bead's growth stopping prematurely at a voxel that's thin but
  not actually a real neck (e.g. surface noise/wobble).
- **Where beads meet**: two beads' growth fronts approaching the same neck
  from opposite sides need to agree that a wall exists there and not double
  count or leave a gap of unclaimed voxels — needs a clear tie-breaking/
  hand-off rule right at the neck itself.
- **Verification plan**: prototype directly against id 2131 (already
  tested, ~97M voxels, known-good result to compare against) and id 4665
  (~4B voxels, currently unprocessable) before trusting this on real data
  more broadly.

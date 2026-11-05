[![CI Status](https://github.com/janelia-cellmap/cellmap-analyze/actions/workflows/tests.yml/badge.svg)](https://github.com/janelia-cellmap/cellmap-analyze/actions/workflows/tests.yml) [![Codecov](https://codecov.io/gh/janelia-cellmap/cellmap-analyze/branch/main/graph/badge.svg)](https://app.codecov.io/gh/janelia-cellmap/cellmap-analyze)

# cellmap-analyze

A suite of Dask-powered tools for processing and analyzing terabyte-scale 3D segmentation datasets. Supports both isotropic and anisotropic voxel sizes.

---

## Features

### Processing Tools

| Tool                     | CLI Command                  | Description                                                 |
| ------------------------ | ---------------------------- | ----------------------------------------------------------- |
| **Connected Components** | `connected-components`       | Threshold predictions, apply masks, and extract connected components. Volume thresholds are in physical units (nm³). |
| **Clean Components**     | `clean-connected-components` | Refine existing segmentations by removing small/large components. |
| **Contact Sites**        | `contact-sites`              | Identify regions where two segmentations are within a configurable physical distance. Handles mismatched voxel sizes by resampling to a common resolution. |
| **Fill Holes**           | `fill-holes`              | Fill interior gaps in segmented volumes.                    |
| **Filter IDs**           | `filter-ids`                 | Exclude unwanted segmentation IDs. Kept IDs are renumbered 1..N in the order given, unless `preserve_ids: true`. |
| **Mutex Watershed**      | `mws`                        | Mutex watershed agglomeration from affinities.              |
| **Label With Mask**      | `label-with-mask`            | Label one dataset with IDs from another.                    |
| **Morphological Operations** | `morphological-operations` | Erosion and dilation of segmented datasets. Processing order across blocks is not guaranteed. |
| **Skeletonize**          | `skeletonize`                | Generate skeletons from segmented objects with optional pruning and simplification. Automatically resamples to isotropic resolution before skeletonization. |
| **Topological Thinning** | `topological-thinning`       | Thin a segmentation blockwise to curve skeletons or, with `medial_surface: true`, medial surfaces (Lee et al. 1994, as COSEM did). The result is voxel for voxel what thinning the whole volume at once gives; objects that touch are thinned as if alone and keep their IDs. |
| **Compute EDT**          | `compute-edt`                | Compute a Euclidean distance transform over a segmentation, for reuse across repeated `split-narrow-bridges` runs. |
| **Split Narrow Bridges** | `split-narrow-bridges`       | Split accidentally-merged blob-like objects (nuclei, cells) at thin necks via EDT-watershed, preserving all original foreground voxels. |
| **Multiscale**           | `multiscale`                 | Add downsampled levels (`s1`, `s2`, …) to an existing zarr dataset for viewing, until a level fits in one chunk. Integer data (labels) is downsampled by mode and floats by mean, on anisotropic data the finer axes are halved first, and each level's OME translation is placed at its voxel centres. |

These tools chain together into a full predictions-to-segmentations pipeline:
start from raw model output (`connected-components` thresholds prediction
probability maps, or `mws` agglomerates affinities) to get an initial
instance segmentation, optionally refine it (`split-narrow-bridges` to
separate accidentally-merged instances, `fill-holes`, `morphological-operations`),
then finalize with `clean-connected-components` and/or `filter-ids`.

Each processing command that writes a volume then adds a multiscale pyramid to
it, as `multiscale` does, so every output is ready to view. Set
`multiscale: false` in its `run-config.yaml` to skip this; `compute-edt` and
`topological-thinning` add one only with `multiscale: true`.

### Analysis Tools

| Tool                | CLI Command                  | Description                                                           |
| ------------------- | ---------------------------- | --------------------------------------------------------------------- |
| **Measurement**     | `measure`                    | Compute metrics (volume, surface area, radius of gyration, bounding box, principal axes) for objects and contact sites. Principal axes give three axis lengths (equivalent-ellipsoid diameters, as in scikit-image's `regionprops`), elongation (longest / second), flatness (second / third), and the longest axis's direction. Supports raw intensity statistics when a raw dataset is provided. |
| **Fit Lines**       | `fit_lines_to_segmentations` | Fit geometric lines to elongated/cylindrical structures.              |
| **Assign to Organelles** | `assign_to_organelles`  | Map segmented objects to organelles based on centers of mass.         |
| **Sheetness**       | `sheetness`                  | Sheetness (sheet- versus tube-like) and thickness of objects such as ER from their medial surfaces, as COSEM measured them: per-voxel sheetness, and per object its volume and surface area by sheetness and its medial surface by sheetness and thickness. |

---

## Anisotropic data

All operations handle anisotropic voxel sizes (e.g. `(8, 8, 32)` nm in ZYX). Physical-unit parameters like `minimum_volume_nm_3`, `contact_distance_nm`, and `gaussian_smoothing_sigma_nm` are automatically converted to the appropriate per-axis voxel units. When two datasets have different voxel sizes, they are resampled onto the same voxel grid, using nearest-neighbor interpolation for labels; raw intensities in `measure` are block-averaged when read at a coarser voxel size.

A `roi` in a run-config is in nm, `["z_start:z_end", "y_start:y_end", "x_start:x_end"]`, and selects the voxels whose centres lie inside it; the output covers just those voxels.

Datasets are read through [worldarray](https://github.com/JaneliaSciComp/worldarray): zarr v2/v3, N5 and Neuroglancer precomputed, from local paths or `s3://`, `gs://` and `http(s)://`. Its README documents where each format places voxel 0. Each array is read with its spatial axes in z, y, x order, so one stored x, y, z (as TensorSwitch writes OME-Zarr v3) is transposed on read. Outputs are written z, y, x, and `multiscale` adds levels in the dataset's own order.

---

## Medial surfaces and sheetness

`topological-thinning` with `medial_surface: true` gives the medial surface
`sheetness` needs; both run on the segmentation's own voxels. For example, for ER:

```yaml
# topological-thinning
segmentation_path: /path/to/segmentations.zarr/er/s0
output_path: /path/to/segmentations.zarr/er_medial_surface
medial_surface: true
```

```yaml
# sheetness
segmentation_path: /path/to/segmentations.zarr/er/s0
medial_surface_path: /path/to/segmentations.zarr/er_medial_surface/s0
output_path: /path/to/analysis.zarr/er_sheetness
csv_path: /path/to/analysis/er_sheetness
```

Thinning runs in passes over blocks read with a halo of `48 * iterations_per_pass + 2`
voxels, so blocks must be at least that big; more iterations per pass read less per
iteration but need bigger blocks. A pass only rethins blocks that changed or have
changes near them. Thinning peels about a voxel from each side per iteration, so an
object takes about its radius in voxels: thin ER (a few voxels) is quick, while
nuclei or cells are better thinned at a coarser resolution.

Sheetness at a medial-surface voxel is COSEM's measure (Descoteaux et al. 2005) of the
Hessian of its object's blurred mask, the largest over COSEM's scales, by default with
Gaussian derivatives from one voxel up to 22.6 voxels (`scale_steps: 8` stops at 5.7,
nine times faster and enough for ER at 8 nm). Sheets of any thickness score about
0.7-0.87, tubes 4 or more voxels wide about 0.12-0.16 and thinner ones up to 0.3.
`hessian: cosem` takes COSEM's own finite differences instead, which score sheets
thinner than 3 voxels only 0.3-0.5. The scales are in voxels, so compare datasets at
the same resolution.
Each other voxel takes the mean sheetness of its nearest medial-surface voxels by
geodesic distance within its object. Thickness is twice a medial-surface voxel's
distance to the nearest voxel outside its object: within half a voxel for sheets at
any angle, but a full voxel high for an axis-aligned sheet an odd number of voxels
thick.

---

## Installation

```bash
pip install cellmap-analyze
```

---

## Usage

All commands share the same basic interface:

```bash
<command> [options] <config_path>
```

* `<command>`: One of the processing or analysis tools listed above.
* `<config_path>`: Directory containing:

  * `run-config.yaml` (parameters for your chosen command)
  * `dask-config.yaml` (Dask cluster settings)

**Options**:

* `-n, --num-workers N`: Number of Dask workers to launch.

> **Output:** A new directory named `config_path-<YYYYMMDDHHMMSS>` will be created, containing copies of your configs and an `output.log` for monitoring.

---

## Configuration Examples
The following run-config.yaml could be used to run `connected-components`.
### run-config.yaml

```yaml
input_path: /path/to/predictions.zarr/mito/s0
output_path: /path/to/segmentations.zarr/mito
intensity_threshold_minimum: 0.71
minimum_volume_nm_3: 1E7
delete_tmp: true
connectivity: 1
mask_config:
  cell:
    path: /path/to/masks.zarr/cell/s0
    mask_type: inclusive
fill_holes: true
```

### dask-config.yaml

The following `dask-config.yaml` files can be used for a variety of tasks.
#### Local

```yaml
jobqueue:
  local:
    ncpus: 1
    processes: 1
    cores: 1
    log-directory: job-logs
    name: dask-worker

distributed:
  scheduler:
    work-stealing: true
```
#### LSF Cluster

```yaml
jobqueue:
  lsf:
    ncpus: 8        # cores per job chunk
    processes: 12  # worker processes per chunk
    cores: 12      # threads per process (1 thread each)
    memory: 120GB  # 15 GB per slot
    walltime: 08:00
    mem: 120000000000
    use-stdin: true
    log-directory: job-logs
    name: cellmap-analyze
    project: charge_group

distributed:
  scheduler:
    work-stealing: true
  admin:
    log-format: '[%(asctime)s] %(levelname)s %(message)s'
    tick:
      interval: 20ms
      limit: 3h
```

Each worker reads and writes with its share of the job's CPUs, `ncpus` (or `cores`) divided by `processes`, at least 1: the example above gives each worker 1 thread. The job script exports it as `WORLDARRAY_THREADS`.

### Submission
To run on 12 dask workers:

**Local run example:**

```bash
connected-components -n 12 config_path
```

**Cluster submit example (LSF):**

```bash
bsub -n 4 -P chargegroup connected-components -n 12 config_path
```

---

## Acknowledgements

The center-finding implementation is taken from [funlib.evaluate](https://github.com/funkelab/funlib.evaluate).

The thinning kernel is adapted from scikit-image's 3D skeletonization (BSD-3-Clause),
with the blockwise scheme, medial-surface mode and sheetness of
[COSEM's segmentation analysis](https://github.com/janelia-cellmap/cosem-segmentation-analysis).

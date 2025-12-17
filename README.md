[![CI Status](https://github.com/janelia-cellmap/cellmap-analyze/actions/workflows/tests.yml/badge.svg)](https://github.com/janelia-cellmap/cellmap-analyze/actions/workflows/tests.yml) [![Codecov](https://codecov.io/gh/janelia-cellmap/cellmap-analyze/branch/main/graph/badge.svg)](https://app.codecov.io/gh/janelia-cellmap/cellmap-analyze)


# cellmap-analyze

A suite of Dask-powered tools for processing and analyzing terabyte-scale 3D segmentation datasets.

---

## Features

### Processing Tools

| Tool                     | CLI Command                  | Description                                                 |
| ------------------------ | ---------------------------- | ----------------------------------------------------------- |
| **Connected Components** | `connected-components`       | Threshold and mask segmentations.                           |
| **Clean Components**     | `clean-connected-components` | Refine and clean existing segmentations.                    |
| **Contact Sites**        | `contact-sites`              | Identify object contact regions with configurable contact distance. |
| **Fill Holes**           | `fill-holes`              | Fill interior gaps in segmented volumes.                    |
| **Filter IDs**           | `filter-ids`                 | Exclude unwanted segmentation IDs.                          |
| **Mutex Watershed**           | `mws`                 | Mutex watershed agglomeration from affinities                          |
| **Label With Mask**           | `label-with-mask`                 | Label one dataset with ids from another                          |
| **Watershed Segmentation**           | `watershed-segmentation`                 | Watershed a segmentation; currently only works by doing watershed globally, but distance transform and seed finding work blockwise|
| **Morphological Operations**           | `morphological-operations`                 | Allowe for erosion and dilation of a segmented dataset, but since ordering of processing may matter, there is no guarantee of consistency based on order of blocks processed|
| **Skeletonize**           | `skeletonize`                 | Generate skeletons from segmented objects with optional pruning and simplification.|

### Analysis Tools

| Tool                | CLI Command                  | Description                                                           |
| ------------------- | ---------------------------- | --------------------------------------------------------------------- |
| **Measurement**     | `measure`                    | Compute metrics (volume, surface area) for objects and contact sites. |
| **Fit Lines**       | `fit_lines_to_segmentations` | Fit geometric lines to elongated/cylindrical structures.              |
| **Assign to Cells** | `assign_to_cells`            | Map segmented objects to cells based on centers of mass.              |

---

## Installation

Install via PyPI:

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
    mem: 12000000000
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

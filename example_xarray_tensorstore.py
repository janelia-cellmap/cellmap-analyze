# %%
# Benchmark: compare read performance across backends
# Uses random locations each iteration to avoid OS page cache effects
import numpy as np
import time
import random as pyrandom

# Path to the dataset
data_path = "/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8/s0"

# %%
# =============================================================================
# 1. Setup
# =============================================================================
import zarr
import tensorstore as ts
import xarray as xr
from cellmap_analyze.util.xarray_image_data_interface import (
    XarrayImageDataInterface,
    Roi,
    Coordinate
)
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from funlib.geometry import Roi as FunlibRoi, Coordinate as FunlibCoordinate

store = zarr.open(data_path, mode='r')
print(f"Shape: {store.shape}")
print(f"Dtype: {store.dtype}")
print(f"Chunks: {store.chunks}")

VOXEL_SIZE = 8  # nm

# Pre-open stores/interfaces
zarr_store = zarr.open(data_path, mode='r')

ts_spec = {
    'driver': 'zarr',
    'kvstore': {'driver': 'file', 'path': data_path},
}
ts_arr = ts.open(ts_spec).result()

ts_spec_single = {
    'driver': 'zarr',
    'kvstore': {'driver': 'file', 'path': data_path},
    'context': {
        'data_copy_concurrency': {'limit': 1},
        'file_io_concurrency': {'limit': 1},
    },
}
ts_arr_single = ts.open(ts_spec_single).result()

idi = XarrayImageDataInterface(data_path, concurrency_limit=1)
idi_original = ImageDataInterface(data_path)

print(f"\nXarrayImageDataInterface metadata:")
print(f"  Voxel size: {idi.voxel_size}")
print(f"  Offset: {idi.offset}")
print(f"\nImageDataInterface (original) metadata:")
print(f"  Voxel size: {idi_original.voxel_size}")
print(f"  Offset: {idi_original.offset}")


# %%
# =============================================================================
# 2. Helper: generate random non-overlapping origins
# =============================================================================
def generate_random_origins(shape, read_size, n):
    """Generate n random voxel origins that fit within shape."""
    origins = []
    for _ in range(n):
        origin = [
            pyrandom.randint(0, s - read_size) for s in shape
        ]
        origins.append(origin)
    return origins


def run_benchmark(name, read_fn, origins, read_size):
    """Run benchmark with different random location each iteration.

    Returns list of times in seconds.
    """
    times = []
    for origin in origins:
        start = time.perf_counter()
        data = read_fn(origin, read_size)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return times


def print_results(name, times, pad=30):
    """Print benchmark results."""
    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    median_ms = np.median(times) * 1000
    print(f"{name:<{pad}} median: {median_ms:8.2f} ms  mean: {mean_ms:8.2f} ms (±{std_ms:.2f} ms)  n={len(times)}")


# %%
# =============================================================================
# 3. Define read functions for each backend
# =============================================================================
def read_zarr(origin, size):
    return zarr_store[
        origin[0]:origin[0]+size,
        origin[1]:origin[1]+size,
        origin[2]:origin[2]+size
    ]

def read_ts_parallel(origin, size):
    return ts_arr[
        origin[0]:origin[0]+size,
        origin[1]:origin[1]+size,
        origin[2]:origin[2]+size
    ].read().result()

def read_ts_single(origin, size):
    return ts_arr_single[
        origin[0]:origin[0]+size,
        origin[1]:origin[1]+size,
        origin[2]:origin[2]+size
    ].read().result()

def read_zarr_xarray(origin, size):
    raw = zarr_store[
        origin[0]:origin[0]+size,
        origin[1]:origin[1]+size,
        origin[2]:origin[2]+size
    ]
    return xr.DataArray(raw, dims=['z', 'y', 'x'])

def read_xarray_idi(origin, size):
    nm_origin = Coordinate([o * VOXEL_SIZE for o in origin])
    nm_size = size * VOXEL_SIZE
    roi = Roi(begin=nm_origin, shape=Coordinate([nm_size, nm_size, nm_size]))
    return idi.to_ndarray_ts(roi)

def read_idi_ts(origin, size):
    nm_origin = FunlibCoordinate([o * VOXEL_SIZE for o in origin])
    nm_size = size * VOXEL_SIZE
    roi = FunlibRoi(nm_origin, FunlibCoordinate([nm_size, nm_size, nm_size]))
    return idi_original.to_ndarray_ts(roi)

def read_idi_ds(origin, size):
    nm_origin = FunlibCoordinate([o * VOXEL_SIZE for o in origin])
    nm_size = size * VOXEL_SIZE
    roi = FunlibRoi(nm_origin, FunlibCoordinate([nm_size, nm_size, nm_size]))
    return idi_original.to_ndarray_ds(roi)


# %%
# =============================================================================
# 4. Warmup pass (separate from timing)
# =============================================================================
print(f"\n{'='*60}")
print("WARMUP (initializing caches, xarray indexes, etc.)")
print(f"{'='*60}")

warmup_origin = [0, 0, 0]
warmup_size = 64

print("  Warming up zarr...")
_ = read_zarr(warmup_origin, warmup_size)
print("  Warming up tensorstore (parallel)...")
_ = read_ts_parallel(warmup_origin, warmup_size)
print("  Warming up tensorstore (single)...")
_ = read_ts_single(warmup_origin, warmup_size)
print("  Warming up zarr+xarray...")
_ = read_zarr_xarray(warmup_origin, warmup_size)
print("  Warming up XarrayImageDataInterface...")
start = time.perf_counter()
_ = read_xarray_idi(warmup_origin, warmup_size)
xidi_init_time = time.perf_counter() - start
print(f"    (first-call init took {xidi_init_time*1000:.1f} ms)")
print("  Warming up ImageDataInterface (ts)...")
_ = read_idi_ts(warmup_origin, warmup_size)
print("  Warming up ImageDataInterface (ds)...")
_ = read_idi_ds(warmup_origin, warmup_size)
print("  Warmup done.\n")


# %%
# =============================================================================
# 5. Benchmark: 256^3 voxels, random locations (no cache reuse)
# =============================================================================
n_runs = 10
READ_SIZE = 256
pyrandom.seed(42)  # reproducible
origins = generate_random_origins(store.shape, READ_SIZE, n_runs)

print(f"{'='*60}")
print(f"BENCHMARK: {READ_SIZE}^3 voxels, {n_runs} random locations")
print(f"{'='*60}\n")

backends = [
    ("Zarr direct", read_zarr),
    ("Tensorstore (parallel)", read_ts_parallel),
    ("Tensorstore (1 thread)", read_ts_single),
    ("zarr+xarray (no dask)", read_zarr_xarray),
    ("XarrayImageDataInterface", read_xarray_idi),
    ("ImageDataInterface (ts)", read_idi_ts),
    ("ImageDataInterface (ds)", read_idi_ds),
]

results_256 = {}
for name, fn in backends:
    times = run_benchmark(name, fn, origins, READ_SIZE)
    results_256[name] = times
    print_results(name, times)

# Data verification: read same location with all backends
verify_origin = origins[0]
ref = read_zarr(verify_origin, READ_SIZE)
print(f"\nData verification (origin={verify_origin}):")
print(f"  zarr == ts (parallel):  {np.array_equal(ref, np.array(read_ts_parallel(verify_origin, READ_SIZE)))}")
print(f"  zarr == ts (1 thread):  {np.array_equal(ref, np.array(read_ts_single(verify_origin, READ_SIZE)))}")
print(f"  zarr == zarr+xarray:    {np.array_equal(ref, read_zarr_xarray(verify_origin, READ_SIZE).values)}")
print(f"  zarr == XarrayIDI:      {np.array_equal(ref, read_xarray_idi(verify_origin, READ_SIZE))}")
print(f"  zarr == IDI (ts):       {np.array_equal(ref, read_idi_ts(verify_origin, READ_SIZE))}")
print(f"  zarr == IDI (ds):       {np.array_equal(ref, read_idi_ds(verify_origin, READ_SIZE))}")


# %%
# =============================================================================
# 6. Benchmark: 512^3 voxels, random locations
# =============================================================================
READ_SIZE_LARGE = 512
pyrandom.seed(99)
origins_large = generate_random_origins(store.shape, READ_SIZE_LARGE, n_runs)

print(f"\n{'='*60}")
print(f"BENCHMARK: {READ_SIZE_LARGE}^3 voxels, {n_runs} random locations")
print(f"{'='*60}\n")

results_512 = {}
for name, fn in backends:
    times = run_benchmark(name, fn, origins_large, READ_SIZE_LARGE)
    results_512[name] = times
    print_results(name, times)


# %%
# =============================================================================
# 7. Summary
# =============================================================================
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"""
Each iteration reads from a DIFFERENT random location to avoid
OS page cache effects. Median is reported alongside mean.

Backends:
- Zarr: Direct array access, sequential I/O
- Tensorstore (parallel): Parallel async I/O (default concurrency)
- Tensorstore (1 thread): Single-threaded (matches XarrayIDI's concurrency_limit=1)
- zarr+xarray (no dask): zarr read + xarray wrapper (measures xarray wrapping cost)
- XarrayImageDataInterface: Physical coordinates, concurrency_limit=1
- ImageDataInterface (ts): Physical coordinates, tensorstore, concurrency_limit=1
- ImageDataInterface (ds): Physical coordinates, funlib.persistence (zarr)

XarrayIDI first-call initialization: {xidi_init_time*1000:.1f} ms
  (builds xarray coordinate indexes for full {store.shape} array)
""")


# %%
# =============================================================================
# 8. Example: output_voxel_size = 2x input voxel size (downsampling)
# =============================================================================
print(f"\n{'='*60}")
print("EXAMPLE: output_voxel_size = 2x input voxel size")
print(f"{'='*60}\n")

# Create an interface with output_voxel_size = 2 * input_voxel_size
# Input voxel size is 8nm, so output will be 16nm (downsampling by 2x)
idi_2x = XarrayImageDataInterface(
    data_path,
    output_voxel_size=Coordinate([VOXEL_SIZE * 2] * 3),
    concurrency_limit=1,
)

print(f"Input voxel size:  {idi_2x.voxel_size}")
print(f"Output voxel size: {idi_2x.output_voxel_size}")

# Read a region at the original resolution and at 2x
origin_nm = Coordinate([0, 0, 0])
size_nm = Coordinate([256 * VOXEL_SIZE] * 3)  # 256 voxels at input resolution
roi = Roi(begin=origin_nm, shape=size_nm)

start = time.perf_counter()
data_1x = idi.to_ndarray_ts(roi)
time_1x = time.perf_counter() - start

start = time.perf_counter()
data_2x = idi_2x.to_ndarray_ts(roi)
time_2x = time.perf_counter() - start

print(f"\nROI: {roi}")
print(f"  1x shape (voxel_size={VOXEL_SIZE}nm): {data_1x.shape}  time: {time_1x*1000:.2f} ms")
print(f"  2x shape (voxel_size={VOXEL_SIZE*2}nm): {data_2x.shape}  time: {time_2x*1000:.2f} ms")
print(f"  Expected 2x shape: {tuple(s // 2 for s in data_1x.shape)}")

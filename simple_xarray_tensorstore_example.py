"""
Simple example: Reading a zarr chunk using xarray + tensorstore.

Uses open_ds_tensorstore for single-threaded tensorstore I/O,
and _TensorStoreAdapter from xarray_tensorstore for lazy xarray wrapping.
"""
import time
import json
import numpy as np
import xarray as xr
import tensorstore as ts
from xarray_tensorstore import _TensorStoreAdapter

# Paths
data_path = "/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8"
array_path = f"{data_path}/s0"


def open_ds_tensorstore(dataset_path, mode="r", concurrency_limit=None):
    filetype = (
        "zarr" if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5") else "n5"
    )
    spec = {
        "driver": filetype,
        "kvstore": {"driver": "file", "path": dataset_path},
    }
    if concurrency_limit:
        spec["context"] = {
            "data_copy_concurrency": {"limit": concurrency_limit},
            "file_io_concurrency": {"limit": concurrency_limit},
        }

    if mode == "r":
        return ts.open(spec, read=True, write=False).result()
    else:
        return ts.open(spec, read=False, write=True).result()


# Open with concurrency_limit=1 (single-threaded)
ts_arr = open_ds_tensorstore(array_path, concurrency_limit=1)

# Read OME-NGFF metadata for dimension names
with open(f"{data_path}/.zattrs") as f:
    zattrs = json.load(f)

axes = [a["name"] for a in zattrs["multiscales"][0]["axes"]]
print(f"Dims: {axes}")
print(f"Shape: {ts_arr.shape}")

# Wrap tensorstore array with xarray (lazy — no data loaded yet)
adapter = _TensorStoreAdapter(ts_arr)
da = xr.DataArray(xr.Variable(axes, adapter), dims=axes)
print(da)

# Read a small chunk (256^3 voxels from origin)
chunk = da[0:256, 0:256, 0:256]
start = time.perf_counter()
data = chunk.values  # triggers actual read via tensorstore
elapsed = time.perf_counter() - start
print(f"\nLoaded chunk shape: {data.shape}")
print(f"Read time: {elapsed*1000:.2f} ms")
print(f"Data min/max: {data.min()}, {data.max()}")

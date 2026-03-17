# %%
"""Script to read individual affinity channel datasets and combine them
into a single multichannel dataset.

Given predictions stored as separate channels:
    {dataset}.zarr/{crop}/mito_aff_0/s0
    {dataset}.zarr/{crop}/mito_aff_1/s0
    {dataset}.zarr/{crop}/mito_aff_2/s0

This script combines them into:
    {output_base}/{dataset}.zarr/{crop}/mito/s0  (shape: 3 x Z x Y x X)
"""

import os
import re
import numpy as np
import csv
import zarr
import tensorstore as ts
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from funlib.persistence import prepare_ds
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.zarr_util import write_multiscales_metadata

# %%
# Fill in these variables
input_base = "/nrs/cellmap/zouinkhim/challenge/v4/predictions"
output_base = "/nrs/cellmap/ackermand/challenge/v4/predictions"  # fill in output path
output_name = "mito_affs"

# %%
# Load crop manifests to check which crops have mito labels
scripts_dir = os.path.dirname(__file__)
crops_with_mito = set()
for manifest_name in ["test_crop_manifest.csv", "train_crop_manifest.csv"]:
    with open(os.path.join(scripts_dir, manifest_name)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "mito" in row["class_label"]:
                crops_with_mito.add(row["crop_name"])

# %%
dataset_dirs = sorted(d for d in os.listdir(input_base) if d.endswith(".zarr"))

for ds_name in dataset_dirs:
    ds_path = os.path.join(input_base, ds_name)
    crops = sorted(
        d for d in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, d))
    )

    for crop in crops:
        # Skip crops that don't have mito in either manifest
        # crop is like "crop557", manifest crop_name is like "557"
        crop_num = crop.replace("crop", "")
        if crop_num not in crops_with_mito:
            print(f"  Skipping {ds_name}/{crop} (no mito in manifests)")
            continue
        crop_path = os.path.join(ds_path, crop)

        # Find aff channels
        entries = sorted(os.listdir(crop_path))
        aff_pattern = re.compile(r"^(.+_aff)_(\d+)$")
        aff_indices = {}
        for entry in entries:
            m = aff_pattern.match(entry)
            if m:
                prefix = m.group(1)
                idx = int(m.group(2))
                aff_indices.setdefault(prefix, []).append(idx)

        for prefix, indices in aff_indices.items():
            indices.sort()
            num_channels = len(indices)
            print(
                f"Processing {ds_name}/{crop}/{prefix} "
                f"({num_channels} channels: {indices})"
            )

            # Read first channel to get metadata
            first_channel_path = os.path.join(crop_path, f"{prefix}_{indices[0]}", "s0")
            first_idi = ImageDataInterface(first_channel_path)
            voxel_size = first_idi.voxel_size
            roi = first_idi.roi
            dtype = first_idi.dtype
            spatial_shape = np.array(first_idi.ds.data.shape)
            chunk_shape = first_idi.chunk_shape

            # Create output dataset
            output_ds_path = os.path.join(output_base, ds_name, crop, output_name)
            os.makedirs(output_ds_path, exist_ok=True)

            write_size = chunk_shape * voxel_size
            ds = prepare_ds(
                os.path.join(output_base, ds_name),
                f"{crop}/{output_name}/s0",
                total_roi=roi,
                write_size=write_size,
                voxel_size=voxel_size,
                dtype=dtype,
                num_channels=num_channels,
                force_exact_write_size=True,
                delete=True,
            )

            # Write spatial-only multiscale metadata
            write_multiscales_metadata(
                output_ds_path,
                "s0",
                list(voxel_size),
                list(roi.get_begin()),
                "nanometer",
                ["z", "y", "x"],
            )

            # Open input tensorstores for each channel
            input_ts_list = []
            for idx in indices:
                channel_path = os.path.join(crop_path, f"{prefix}_{idx}", "s0")
                input_ts_list.append(
                    ts.open(
                        {
                            "driver": "zarr",
                            "kvstore": {"driver": "file", "path": channel_path},
                        },
                        read=True,
                    ).result()
                )

            # Open output tensorstore
            output_zarr_path = os.path.join(output_base, ds_name)
            output_ts = ts.open(
                {
                    "driver": "zarr",
                    "kvstore": {"driver": "file", "path": output_zarr_path},
                    "path": f"{crop}/{output_name}/s0",
                },
                open=True,
            ).result()

            # Read and write chunks in parallel
            cs = np.array(chunk_shape)
            z_ranges = range(0, spatial_shape[0], cs[0])
            y_ranges = range(0, spatial_shape[1], cs[1])
            x_ranges = range(0, spatial_shape[2], cs[2])
            total_chunks = len(z_ranges) * len(y_ranges) * len(x_ranges)

            def process_chunk(z0, y0, x0):
                z1 = min(z0 + cs[0], spatial_shape[0])
                y1 = min(y0 + cs[1], spatial_shape[1])
                x1 = min(x0 + cs[2], spatial_shape[2])
                chunk_channels = []
                for in_ts in input_ts_list:
                    chunk_data = in_ts[z0:z1, y0:y1, x0:x1].read().result()
                    chunk_channels.append(chunk_data)
                chunk_combined = np.stack(chunk_channels, axis=0)
                output_ts[:, z0:z1, y0:y1, x0:x1].write(chunk_combined).result()

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [
                    executor.submit(process_chunk, z0, y0, x0)
                    for z0, y0, x0 in product(z_ranges, y_ranges, x_ranges)
                ]
                for f in tqdm(
                    as_completed(futures),
                    total=total_chunks,
                    desc=f"  {crop}/{prefix}",
                ):
                    f.result()  # raise any exceptions

            print(
                f"  Written to {output_ds_path} (shape: {num_channels}x{list(spatial_shape)})"
            )

print("Done!")

# %%

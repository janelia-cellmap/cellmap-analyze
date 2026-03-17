# %%
"""Create MutexWatershed run configs for each dataset/crop combo that has mito affinities.

For each valid dataset/crop, creates:
    {config_base}/{dataset}/{crop}/mito/dask-config.yaml  (copied from template)
    {config_base}/{dataset}/{crop}/mito/run-config.yaml   (paths updated per dataset/crop)
"""

import os
import csv
import re
import shutil
import yaml

# %%
# Fill in these variables
input_base = "/nrs/cellmap/ackermand/challenge/v4/predictions"
output_base = "/nrs/cellmap/ackermand/challenge/v4/processed"
config_base = "/groups/cellmap/cellmap/ackermand/cellmap-analyze-scripts/segmentation-challenge"
template_dir = os.path.join(config_base, "jrc_ctl-id8-1", "crop557", "template_mito")

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
# Read template configs
with open(os.path.join(template_dir, "run-config.yaml")) as f:
    template_run_config = yaml.safe_load(f)

dataset_dirs = sorted(d for d in os.listdir(input_base) if d.endswith(".zarr"))

for ds_name in dataset_dirs:
    ds_path = os.path.join(input_base, ds_name)
    dataset = ds_name.replace(".zarr", "")
    crops = sorted(
        d for d in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, d))
    )

    for crop in crops:
        crop_num = crop.replace("crop", "")
        if crop_num not in crops_with_mito:
            continue

        # Check that mito_affs actually exists for this crop
        affs_path = os.path.join(ds_path, crop, "mito_affs", "s0")
        if not os.path.exists(affs_path):
            print(f"  Skipping {ds_name}/{crop} (no mito_affs/s0 found)")
            continue

        # Create config directory
        config_dir = os.path.join(config_base, dataset, crop, "mito")
        os.makedirs(config_dir, exist_ok=True)

        # Copy dask-config.yaml as-is
        shutil.copy2(
            os.path.join(template_dir, "dask-config.yaml"),
            os.path.join(config_dir, "dask-config.yaml"),
        )

        # Create run-config.yaml with updated paths
        run_config = dict(template_run_config)
        run_config["affinities_path"] = os.path.join(
            input_base, ds_name, crop, "mito_affs", "s0"
        )
        run_config["output_path"] = os.path.join(
            output_base, f"{dataset}.zarr", crop, "mito"
        )

        with open(os.path.join(config_dir, "run-config.yaml"), "w") as f:
            yaml.dump(run_config, f, default_flow_style=False, sort_keys=False)

        print(f"  Created {config_dir}")

print("Done!")

# %%

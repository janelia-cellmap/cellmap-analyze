# %%
"""Submit MutexWatershed bsub jobs for each dataset/crop combo that has mito configs."""

import os
import csv
import subprocess

# %%
# Fill in these variables
input_base = "/nrs/cellmap/ackermand/challenge/v4/predictions"
config_base = (
    "/groups/cellmap/cellmap/ackermand/cellmap-analyze-scripts/segmentation-challenge"
)
ncpus = 48

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
    dataset = ds_name.replace(".zarr", "")
    crops = sorted(
        d for d in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, d))
    )

    for crop in crops:
        crop_num = crop.replace("crop", "")
        if crop_num not in crops_with_mito:
            continue

        config_dir = os.path.join(config_base, dataset, crop, "mito")
        if not os.path.exists(os.path.join(config_dir, "run-config.yaml")):
            print(f"  Skipping {dataset}/{crop} (no config found)")
            continue

        cmd = [
            "bsub",
            "-n",
            str(ncpus),
            "-P",
            "cellmap",
            "mutex-watershed",
            config_dir,
            "-n",
            str(ncpus),
        ]
        print(f"  Submitting {dataset}/{crop}: {' '.join(cmd)}")
        subprocess.run(cmd)

print("Done!")

# %%

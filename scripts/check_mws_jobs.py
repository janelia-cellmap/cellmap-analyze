"""Check that the latest mito-* run for each crop completed successfully."""

import os
import sys

config_base = (
    "/groups/cellmap/cellmap/ackermand/cellmap-analyze-scripts/segmentation-challenge"
)

failed = []
for dataset in sorted(os.listdir(config_base)):
    dataset_dir = os.path.join(config_base, dataset)
    if not os.path.isdir(dataset_dir):
        continue
    for crop in sorted(os.listdir(dataset_dir)):
        crop_dir = os.path.join(dataset_dir, crop)
        if not os.path.isdir(crop_dir):
            continue

        # Find all mito-<datetime> directories (exclude plain "mito" config dir)
        mito_runs = sorted(
            d for d in os.listdir(crop_dir)
            if d.startswith("mito-") and os.path.isdir(os.path.join(crop_dir, d))
        )
        if not mito_runs:
            failed.append((dataset, crop, "no mito-* run directories found"))
            continue

        latest = mito_runs[-1]  # lexicographic sort on datetime works
        output_log = os.path.join(crop_dir, latest, "output.log")
        if not os.path.isfile(output_log):
            failed.append((dataset, crop, f"{latest}: output.log not found"))
            continue

        with open(output_log) as f:
            lines = f.readlines()

        if not lines:
            failed.append((dataset, crop, f"{latest}: output.log is empty"))
            continue

        last_line = lines[-1].strip()
        if "Complete success mutexwatershed" not in last_line:
            failed.append((dataset, crop, f"{latest}: {last_line}"))
            continue

        print(f"OK   {dataset}/{crop} ({latest})")

if failed:
    print()
    for dataset, crop, reason in failed:
        print(f"FAIL {dataset}/{crop}: {reason}")
    sys.exit(1)
else:
    print("\nAll crops completed successfully.")

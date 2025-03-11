from cellmap_analyze.analyze.fit_lines_to_segmentation import FitLinesToSegmentations

import numpy as np
import pandas as pd


def test_fit_lines_to_segmentations(
    shared_tmpdir,
    tmp_zarr,
    tmp_cylinders_information_csv,
    horizontal_cylinder_endpoints,
    vertical_cylinder_endpoints,
    diagonal_cylinder_endpoints,
):

    cc = FitLinesToSegmentations(
        input_csv=tmp_cylinders_information_csv,
        segmentation_ds_path=f"{tmp_zarr}/segmentation_cylinders/s0",
        output_annotations_dir=f"{shared_tmpdir}/annotations/",
        num_workers=1,
    )
    cc.get_fit_lines_to_segmentations()
    df = pd.read_csv(tmp_cylinders_information_csv.replace(".csv", "_lines.csv"))

    fit_properly = True
    # should be in order 1,2,3 which is horizontal, vertical and diagonal cylinders
    for i, endpoints in enumerate(
        [
            horizontal_cylinder_endpoints,
            vertical_cylinder_endpoints,
            diagonal_cylinder_endpoints,
        ]
    ):
        row = df.iloc[i]
        start = np.array([row[f"Line Start {d} (nm)"] for d in ["Z", "Y", "X"]])
        end = np.array([row[f"Line End {d} (nm)"] for d in ["Z", "Y", "X"]])
        fit_properly &= np.allclose(
            np.vstack([start, end]),
            endpoints,
        ) or np.allclose(
            np.vstack([end, start]),
            endpoints,
        )
    assert fit_properly

"""Two operations sharing an output_path must not collide on a fixed-name
tmp/merge directory. Each construction gets a per-instance UUID suffix so
concurrent jobs can't stomp on each other's intermediate files.

This test asserts the suffix is wired in for every op that previously had a
fixed-name merge dir; it does not exercise concurrency directly, but the
distinct ``_run_id`` is the structural guarantee that makes concurrent
output_path sharing safe.
"""
import os
import re

import numpy as np
import pandas as pd
import pytest
import zarr
from funlib.geometry import Coordinate, Roi

from cellmap_analyze.process.skeletonize import Skeletonize
from cellmap_analyze.process.connected_components import ConnectedComponents
from cellmap_analyze.process.clean_connected_components import (
    CleanConnectedComponents,
)
from cellmap_analyze.process.fill_holes import FillHoles
from cellmap_analyze.analyze.assign_to_organelles import AssignToOrganelles
from cellmap_analyze.util.zarr_util import create_multiscale_dataset


HEX8 = re.compile(r"^[0-9a-f]{8}$")


def _make_seg(path, shape=(8, 8, 8), vs=(8, 8, 8)):
    """Helper: write a small labelled zarr."""
    total_roi = Roi(Coordinate(0, 0, 0), Coordinate(shape) * Coordinate(vs))
    ds = create_multiscale_dataset(
        path,
        dtype=np.uint8,
        voxel_size=list(vs),
        total_roi=total_roi,
        write_size=Coordinate(shape) * Coordinate(vs),
        original_voxel_size=list(vs),
    )
    arr = np.zeros(shape, dtype=np.uint8)
    arr[2:5, 2:5, 2:5] = 1
    ds.data[:] = arr
    return f"{path}/s0"


def _make_bbox_csv(path):
    pd.DataFrame(
        [
            {
                "Object ID": 1,
                "MIN Z (nm)": 20.0,
                "MIN Y (nm)": 20.0,
                "MIN X (nm)": 20.0,
                "MAX Z (nm)": 36.0,
                "MAX Y (nm)": 36.0,
                "MAX X (nm)": 36.0,
            }
        ]
    ).set_index("Object ID").to_csv(path)


def _make_coms_csv(path):
    pd.DataFrame(
        {
            "Object ID": [1],
            "COM X (nm)": [28.0],
            "COM Y (nm)": [28.0],
            "COM Z (nm)": [28.0],
        }
    ).to_csv(path, index=False)


def test_skeletonize_run_id_unique(tmp_path):
    seg = _make_seg(str(tmp_path / "seg.zarr/seg"))
    csv = str(tmp_path / "bboxes.csv")
    _make_bbox_csv(csv)
    out = str(tmp_path / "skel_out")
    a = Skeletonize(
        segmentation_path=seg, output_path=out, csv_path=csv, sharded=False,
        num_workers=1,
    )
    b = Skeletonize(
        segmentation_path=seg, output_path=out, csv_path=csv, sharded=False,
        num_workers=1,
    )
    assert a._run_id != b._run_id
    assert HEX8.match(a._run_id) and HEX8.match(b._run_id)


def test_connected_components_run_id_unique(tmp_path):
    seg = _make_seg(str(tmp_path / "cc_in.zarr/seg"))
    out = str(tmp_path / "cc_out.zarr/labels")
    a = ConnectedComponents(
        input_path=seg, output_path=out, intensity_threshold_minimum=1,
        num_workers=1,
    )
    b = ConnectedComponents(
        input_path=seg, output_path=out, intensity_threshold_minimum=1,
        num_workers=1,
    )
    assert a._run_id != b._run_id


def test_clean_connected_components_run_id_unique(tmp_path):
    seg = _make_seg(str(tmp_path / "ccc_in.zarr/seg"))
    out = str(tmp_path / "ccc_out.zarr/labels")
    a = CleanConnectedComponents(input_path=seg, output_path=out, num_workers=1)
    b = CleanConnectedComponents(input_path=seg, output_path=out, num_workers=1)
    assert a._run_id != b._run_id


def test_fill_holes_run_id_unique(tmp_path):
    seg = _make_seg(str(tmp_path / "fh_in.zarr/seg"))
    out = str(tmp_path / "fh_out.zarr/labels")
    a = FillHoles(input_path=seg, output_path=out, num_workers=1)
    b = FillHoles(input_path=seg, output_path=out, num_workers=1)
    assert a._run_id != b._run_id


def test_assign_to_organelles_run_id_unique(tmp_path):
    target = _make_seg(str(tmp_path / "target.zarr/nuc"))
    coms = str(tmp_path / "coms.csv")
    _make_coms_csv(coms)
    out = str(tmp_path / "assign_out")
    a = AssignToOrganelles(
        organelle_csvs=coms,
        target_organelle_ds_path=target,
        output_path=out,
        num_workers=1,
    )
    b = AssignToOrganelles(
        organelle_csvs=coms,
        target_organelle_ds_path=target,
        output_path=out,
        num_workers=1,
    )
    assert a._run_id != b._run_id


def test_run_ids_appear_in_tmp_paths_assign(tmp_path):
    """The _run_id is actually used to scope ``.tmp_assign*`` paths."""
    target = _make_seg(str(tmp_path / "target.zarr/nuc"))
    coms = str(tmp_path / "coms.csv")
    _make_coms_csv(coms)
    out = str(tmp_path / "assign_out")
    a = AssignToOrganelles(
        organelle_csvs=coms,
        target_organelle_ds_path=target,
        output_path=out,
        num_workers=1,
    )
    # _save_coms_to_tmp builds the directory path from self._run_id
    coms_path = a._save_coms_to_tmp(np.array([[1.0, 1.0, 1.0]]))
    assert a._run_id in coms_path
    assert ".tmp_assign_" + a._run_id in coms_path

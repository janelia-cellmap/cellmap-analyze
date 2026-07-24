import numpy as np
from funlib.geometry import Coordinate, Roi

from cellmap_analyze.process.split_narrow_bridges import SplitNarrowBridges
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.zarr_util import create_multiscale_dataset


def _write_segmentation(path, seg, voxel_size=(1.0, 1.0, 1.0)):
    total_roi = Roi((0, 0, 0), Coordinate(seg.shape) * Coordinate(voxel_size))
    ds = create_multiscale_dataset(
        path,
        dtype=seg.dtype,
        voxel_size=voxel_size,
        total_roi=total_roi,
        write_size=Coordinate(10, 10, 10) * Coordinate(voxel_size),
        original_voxel_size=voxel_size,
    )
    ds.data[:] = seg
    return f"{path}/s0"


def _dumbbell_segmentation():
    seg = np.zeros((20, 20, 20), dtype=np.uint32)
    # Two 6x6x6 cubes joined by a thin (2x2x1) bridge -- a classic "narrow
    # neck" merge artifact.
    seg[2:8, 2:8, 2:8] = 5
    seg[8:9, 4:6, 4:6] = 5
    seg[9:15, 2:8, 2:8] = 5
    # A separate, untouched small cube -- should never be split.
    seg[2:5, 12:15, 12:15] = 7
    return seg


def test_split_narrow_bridges_edt_watershed(tmp_path):
    seg = _dumbbell_segmentation()
    seg_path = _write_segmentation(str(tmp_path / "segmentation.zarr"), seg)
    output_path = str(tmp_path / "split_output.zarr")

    snb = SplitNarrowBridges(
        segmentation_path=seg_path,
        output_path=output_path,
        strategy="edt_watershed",
        neck_radius_nm=2,
        minimum_subregion_volume_nm_3=None,
        num_workers=1,
    )
    snb.split_objects()

    output = ImageDataInterface(f"{output_path}/s0").to_ndarray_ts()

    # Every original foreground voxel is still foreground, and vice versa --
    # only instance labels may change, never the mask.
    assert np.array_equal(output > 0, seg > 0)

    # The dumbbell (originally a single object, id 5) was split into >= 2
    # new instance labels.
    dumbbell_labels = np.unique(output[seg == 5])
    assert len(dumbbell_labels) >= 2

    # New labels for split pieces never collide with any untouched original id.
    assert np.all(dumbbell_labels > seg.max())

    # The untouched control cube (id 7) keeps its original label untouched.
    assert np.array_equal(output[seg == 7], seg[seg == 7])

    # No voxel outside the dumbbell/control objects picked up a new label.
    other_mask = (seg == 0)
    assert np.all(output[other_mask] == 0)


def test_split_narrow_bridges_no_split_when_disconnected_neck(tmp_path):
    # A single compact cube (no bridge) should never be split.
    seg = np.zeros((20, 20, 20), dtype=np.uint32)
    seg[2:8, 2:8, 2:8] = 3
    seg_path = _write_segmentation(str(tmp_path / "segmentation.zarr"), seg)
    output_path = str(tmp_path / "split_output.zarr")

    snb = SplitNarrowBridges(
        segmentation_path=seg_path,
        output_path=output_path,
        strategy="edt_watershed",
        neck_radius_nm=2,
        minimum_subregion_volume_nm_3=None,
        num_workers=1,
    )
    snb.split_objects()

    output = ImageDataInterface(f"{output_path}/s0").to_ndarray_ts()
    assert np.array_equal(output, seg)

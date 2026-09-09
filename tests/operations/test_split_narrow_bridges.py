import glob

import numpy as np
from funlib.geometry import Coordinate, Roi

from cellmap_analyze.process.compute_edt import ComputeEDT
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


def test_split_narrow_bridges_minimum_object_volume_to_split_gate(tmp_path):
    # minimum_object_volume_to_split_nm_3 is a pre-filter on the whole
    # object, checked before any candidate cut is even looked for -- distinct
    # from minimum_subregion_volume_nm_3, which only gates individual cuts
    # once splitting is already underway. The dumbbell (436 voxels/nm^3
    # total) would normally split under these exact settings (see
    # test_split_narrow_bridges_edt_watershed); a gate set above its total
    # volume should skip the attempt entirely and leave it untouched.
    seg = _dumbbell_segmentation()
    seg_path = _write_segmentation(str(tmp_path / "segmentation.zarr"), seg)
    output_path = str(tmp_path / "split_output.zarr")

    snb = SplitNarrowBridges(
        segmentation_path=seg_path,
        output_path=output_path,
        strategy="edt_watershed",
        neck_radius_nm=2,
        minimum_subregion_volume_nm_3=None,
        minimum_object_volume_to_split_nm_3=500,
        num_workers=1,
    )
    assert 5 not in snb.ids  # filtered out before any object is even read
    snb.split_objects()

    output = ImageDataInterface(f"{output_path}/s0").to_ndarray_ts()

    # Nothing changed: the dumbbell (id 5) and control cube (id 7, always
    # below the gate) both stay exactly as they were.
    assert np.array_equal(output, seg)


def test_split_narrow_bridges_minimum_object_volume_to_split_preserves_new_id_space(
    tmp_path,
):
    # A regression guard for the id-collision trap: new split ids are
    # assigned starting at max(original id) + 1. If that max were computed
    # from the post-filter self.ids (only objects actually attempted) rather
    # than the full bbox table, a filtered-out object with a higher id than
    # every split candidate could collide with a newly assigned split id.
    seg = _dumbbell_segmentation()
    # Give the untouched control cube a higher id than the dumbbell, and set
    # the gate to filter the control cube out (its volume, 27, is tiny) while
    # still letting the dumbbell split.
    seg[seg == 7] = 50
    seg_path = _write_segmentation(str(tmp_path / "segmentation.zarr"), seg)
    output_path = str(tmp_path / "split_output.zarr")

    snb = SplitNarrowBridges(
        segmentation_path=seg_path,
        output_path=output_path,
        strategy="edt_watershed",
        neck_radius_nm=2,
        minimum_subregion_volume_nm_3=None,
        minimum_object_volume_to_split_nm_3=100,
        num_workers=1,
    )
    assert 50 not in snb.ids
    assert 5 in snb.ids
    snb.split_objects()

    output = ImageDataInterface(f"{output_path}/s0").to_ndarray_ts()

    assert np.array_equal(output > 0, seg > 0)
    # The control cube keeps its original (higher) id untouched.
    assert np.array_equal(output[seg == 50], seg[seg == 50])
    # The dumbbell split into new ids that don't collide with id 50.
    dumbbell_labels = np.unique(output[seg == 5])
    assert len(dumbbell_labels) >= 2
    assert not (set(dumbbell_labels.tolist()) & {50})


def test_split_narrow_bridges_final_volume_filter(tmp_path):
    # minimum_volume_nm_3 is a final dataset-level filter, distinct from
    # minimum_subregion_volume_nm_3 (which only gates split acceptance): it
    # should prune a small standalone object that was never even a split
    # candidate, while leaving the dumbbell's split pieces and the larger
    # control cube untouched.
    seg = _dumbbell_segmentation()
    seg[16:18, 16:18, 16:18] = 6  # 2x2x2 = 8 voxels/nm^3 -- below the gate
    seg_path = _write_segmentation(str(tmp_path / "segmentation.zarr"), seg)
    output_path = str(tmp_path / "split_output.zarr")

    snb = SplitNarrowBridges(
        segmentation_path=seg_path,
        output_path=output_path,
        strategy="edt_watershed",
        neck_radius_nm=2,
        minimum_subregion_volume_nm_3=None,
        minimum_volume_nm_3=20,
        num_workers=1,
    )
    snb.split_objects()

    output = ImageDataInterface(f"{output_path}/s0").to_ndarray_ts()

    # The tiny 8-voxel object was pruned entirely.
    assert np.all(output[seg == 6] == 0)
    # The dumbbell still split into >= 2 pieces, each well above the gate.
    # (CleanConnectedComponents renumbers the whole dataset sequentially as
    # part of its final relabel, so split pieces are no longer guaranteed to
    # sit above the original max id -- only distinctness matters here.)
    dumbbell_labels = np.unique(output[seg == 5])
    assert len(dumbbell_labels) >= 2
    assert 0 not in dumbbell_labels
    # The control cube (27 voxels, above the gate) survives as a single object.
    assert len(np.unique(output[seg == 7])) == 1
    assert np.all(output[seg == 7] != 0)
    # No scratch/unfiltered intermediate dataset left behind.
    assert not glob.glob(str(tmp_path / "*_unfiltered_*"))


def _two_scale_dumbbell_segmentation():
    seg = np.zeros((50, 50, 50), dtype=np.uint32)
    # Small dumbbell (id 5): two 6x6x6 cubes joined by a 2x2x1 bridge.
    seg[2:8, 2:8, 2:8] = 5
    seg[8:9, 4:6, 4:6] = 5
    seg[9:15, 2:8, 2:8] = 5
    # Large dumbbell (id 6): same proportions, scaled 2x -- two 12x12x12
    # cubes joined by a 4x4x2 bridge. Its bridge has a proportionally
    # identical (but absolutely larger) EDT radius, so a single fixed
    # neck_radius_nm tuned to accept the small bridge as "thin" rejects the
    # large one as "too thick" -- the doc's motivating "a small vs. huge
    # nucleus don't share a natural scale" problem.
    seg[20:32, 2:14, 2:14] = 6
    seg[32:34, 5:9, 5:9] = 6
    seg[34:46, 2:14, 2:14] = 6
    return seg


def test_split_narrow_bridges_adaptive_neck_radius(tmp_path):
    seg = _two_scale_dumbbell_segmentation()
    seg_path = _write_segmentation(str(tmp_path / "segmentation.zarr"), seg)

    fixed_output_path = str(tmp_path / "split_output_fixed.zarr")
    SplitNarrowBridges(
        segmentation_path=seg_path,
        output_path=fixed_output_path,
        strategy="edt_watershed",
        neck_radius_nm=2,
        neck_radius_mode="fixed",
        minimum_subregion_volume_nm_3=None,
        max_pieces_per_object=200,
        num_workers=1,
    ).split_objects()
    fixed_output = ImageDataInterface(f"{fixed_output_path}/s0").to_ndarray_ts()

    # The fixed threshold (tuned to accept the small dumbbell's thin bridge)
    # splits the small one but under-splits the large one.
    assert len(np.unique(fixed_output[seg == 5])) >= 2
    assert len(np.unique(fixed_output[seg == 6])) == 1

    adaptive_output_path = str(tmp_path / "split_output_adaptive.zarr")
    SplitNarrowBridges(
        segmentation_path=seg_path,
        output_path=adaptive_output_path,
        strategy="edt_watershed",
        neck_radius_nm=2,
        neck_radius_mode="adaptive",
        minimum_subregion_volume_nm_3=None,
        max_pieces_per_object=200,
        num_workers=1,
    ).split_objects()
    adaptive_output = ImageDataInterface(f"{adaptive_output_path}/s0").to_ndarray_ts()

    # Adaptive mode derives each object's own threshold from its own EDT
    # values, so both dumbbells split despite the size mismatch.
    assert len(np.unique(adaptive_output[seg == 5])) >= 2
    assert len(np.unique(adaptive_output[seg == 6])) >= 2
    # Still a verbatim voxel-preserving relabel, same invariant as every
    # other split path.
    assert np.array_equal(adaptive_output > 0, seg > 0)


def test_split_narrow_bridges_uses_precomputed_edt(tmp_path):
    # edt_path (manually supplied) and precompute_edt=True (explicit --
    # False is now the default, since ComputeEDT's fixed-block-grid
    # padding has no memory-aware wave planning of its own and can OOM a
    # dask slot regardless of any single object's size; see
    # docs/split_narrow_bridges_plan.md) should both give identical
    # results to the default per-object edt.edt(mask, ...) recompute.
    seg = _dumbbell_segmentation()
    seg_path = _write_segmentation(str(tmp_path / "segmentation.zarr"), seg)

    baseline_output_path = str(tmp_path / "split_output_baseline.zarr")
    SplitNarrowBridges(
        segmentation_path=seg_path,
        output_path=baseline_output_path,
        strategy="edt_watershed",
        neck_radius_nm=2,
        minimum_subregion_volume_nm_3=None,
        # precompute_edt defaults to False -- left implicit here on purpose,
        # to also cover the default not being passed explicitly.
        num_workers=1,
    ).split_objects()
    baseline = ImageDataInterface(f"{baseline_output_path}/s0").to_ndarray_ts()

    auto_output_path = str(tmp_path / "split_output_auto.zarr")
    SplitNarrowBridges(
        segmentation_path=seg_path,
        output_path=auto_output_path,
        strategy="edt_watershed",
        neck_radius_nm=2,
        minimum_subregion_volume_nm_3=None,
        precompute_edt=True,  # auto-runs ComputeEDT once, mirroring
        # csv_path=None auto-running Measure.
        num_workers=1,
    ).split_objects()
    auto = ImageDataInterface(f"{auto_output_path}/s0").to_ndarray_ts()
    assert np.array_equal(baseline, auto)
    # The auto-computed EDT scratch dataset is cleaned up (delete_tmp
    # defaults to True), same as the split scratch dir.
    assert not glob.glob(str(tmp_path / "*_edt_scratch_*"))

    edt_output_path = str(tmp_path / "edt.zarr")
    ComputeEDT(
        segmentation_path=seg_path,
        output_path=edt_output_path,
        padding_nm=8,
        block_shape_voxels=(10, 10, 10),
        num_workers=1,
    ).calculate_edt()

    wired_output_path = str(tmp_path / "split_output_wired.zarr")
    SplitNarrowBridges(
        segmentation_path=seg_path,
        output_path=wired_output_path,
        strategy="edt_watershed",
        neck_radius_nm=2,
        minimum_subregion_volume_nm_3=None,
        edt_path=f"{edt_output_path}/s0",
        num_workers=1,
    ).split_objects()
    wired = ImageDataInterface(f"{wired_output_path}/s0").to_ndarray_ts()

    assert np.array_equal(baseline, wired)


def test_load_subpieces_cached_respects_byte_budget(tmp_path):
    from collections import OrderedDict

    def _write_scratch(name, num_bytes):
        path = str(tmp_path / f"{name}.npz")
        arr = np.ones(num_bytes, dtype=np.uint8)
        np.savez(path, subpieces=arr)
        return path, arr

    small_a, arr_a = _write_scratch("a", 100)
    small_b, arr_b = _write_scratch("b", 100)
    huge, arr_huge = _write_scratch("huge", 1000)

    cache = OrderedDict()
    max_bytes = 250  # fits both small arrays but not alongside the huge one

    loaded_a = SplitNarrowBridges._load_subpieces_cached(small_a, cache, max_bytes)
    assert np.array_equal(loaded_a, arr_a)
    assert list(cache.keys()) == [small_a]

    loaded_b = SplitNarrowBridges._load_subpieces_cached(small_b, cache, max_bytes)
    assert np.array_equal(loaded_b, arr_b)
    # both small arrays fit together under the budget
    assert list(cache.keys()) == [small_a, small_b]

    # a single object larger than the whole budget is still returned
    # correctly, but doesn't get retained (would otherwise permanently
    # wedge the cache over budget)
    loaded_huge = SplitNarrowBridges._load_subpieces_cached(huge, cache, max_bytes)
    assert np.array_equal(loaded_huge, arr_huge)
    assert huge not in cache
    # loading the oversized entry evicted everything else to make room
    # (even though it ultimately wasn't kept)
    assert list(cache.keys()) == []

    # re-populate, then confirm re-touching an entry (move_to_end) protects
    # it from eviction ahead of a colder one
    SplitNarrowBridges._load_subpieces_cached(small_a, cache, max_bytes)
    SplitNarrowBridges._load_subpieces_cached(small_b, cache, max_bytes)
    SplitNarrowBridges._load_subpieces_cached(small_a, cache, max_bytes)  # re-touch a

    third, arr_third = _write_scratch("c", 100)
    # adding a third 100-byte entry exceeds the 250-byte budget by itself
    # only when all three are present -- b (least recently touched) should
    # be evicted, not a (recently re-touched).
    SplitNarrowBridges._load_subpieces_cached(third, cache, max_bytes)
    assert small_b not in cache
    assert small_a in cache
    assert third in cache

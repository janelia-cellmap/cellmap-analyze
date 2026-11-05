import numpy as np
import pytest
from scipy import ndimage
from skimage.measure import euler_number, label

from cellmap_analyze.cythonizing import topological_thinning as tt
from cellmap_analyze.process.topological_thinning import (
    TopologicalThinning,
    blocks_near_deletions,
    deletion_bands,
)
from cellmap_analyze.util.grid_util import make_grid
from cellmap_analyze.util.zarr_util import create_multiscale_dataset

VOXEL_SIZE = (8.0, 8.0, 8.0)
SHAPE = (126, 118, 131)


def _objects(shape, count, rng, num_labels=5):
    """Blobs and slabs with a few labels, so objects of different labels touch."""
    zz, yy, xx = np.indices(shape)
    labels = np.zeros(shape, np.uint64)
    for _ in range(count):
        c = rng.uniform(0, shape)
        if rng.random() < 0.5:
            r = rng.uniform(3, 14, 3)
            m = (((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2) <= 1
        else:
            n = rng.normal(size=3)
            n /= np.linalg.norm(n)
            d = (zz - c[0]) * n[0] + (yy - c[1]) * n[1] + (xx - c[2]) * n[2]
            m = (np.abs(d) <= rng.uniform(1, 4)) & (
                (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= rng.uniform(10, 40) ** 2
            )
        labels[m] = rng.integers(1, num_labels + 1)
    return labels


def _whole_volume(labels, medial_surface, origin=(0, 0, 0)):
    """Thinning of the whole zero-padded volume, ``origin`` the global index of
    its first voxel."""
    padded = np.pad(labels, 1)
    tt.thin(padded, tuple(int(o) - 1 for o in origin), medial_surface, 0)
    return padded[1:-1, 1:-1, 1:-1]


@pytest.fixture(scope="module")
def segmentation(tmp_path_factory):
    labels = _objects(SHAPE, 60, np.random.default_rng(0))
    path = f"{tmp_path_factory.mktemp('thinning')}/in.zarr/seg"
    vol = create_multiscale_dataset(
        path, dtype=np.uint64, grid=make_grid(SHAPE, VOXEL_SIZE, (16, 24, 40), chunks=(13, 13, 13))
    )
    vol.write_index((0, 0, 0), SHAPE, labels)
    return path, labels


def test_simple_point_is_one_component_without_the_centre():
    rng = np.random.default_rng(1)
    structure = np.ones((3, 3, 3), bool)
    for _ in range(20000):
        neighbourhood = rng.random(27) < rng.uniform(0.1, 0.9)
        neighbourhood[13] = True
        without_centre = neighbourhood.reshape(3, 3, 3).copy()
        without_centre[1, 1, 1] = False
        components = ndimage.label(without_centre, structure=structure)[1]
        assert tt._is_simple_point(neighbourhood) == (components <= 1)


def test_euler_invariance_matches_the_euler_number():
    rng = np.random.default_rng(2)
    for _ in range(3000):
        neighbourhood = rng.random(27) < rng.uniform(0.1, 0.9)
        neighbourhood[13] = True
        with_centre = np.pad(neighbourhood.reshape(3, 3, 3), 1)
        without_centre = with_centre.copy()
        without_centre[2, 2, 2] = False
        invariant = euler_number(with_centre, connectivity=3) == euler_number(
            without_centre, connectivity=3
        )
        assert tt._is_euler_invariant(neighbourhood) == invariant


@pytest.mark.parametrize("medial_surface", [False, True])
def test_thinning_keeps_topology(medial_surface):
    rng = np.random.default_rng(3)
    for _ in range(10):
        binary = (_objects((40, 40, 40), 20, rng, num_labels=1) > 0).astype(np.uint8)
        thinned = _whole_volume(binary, medial_surface)
        assert thinned.sum() < binary.sum()
        assert label(thinned, connectivity=3).max() == label(binary, connectivity=3).max()
        assert euler_number(thinned, connectivity=3) == euler_number(binary, connectivity=3)


@pytest.mark.parametrize("medial_surface", [False, True])
def test_touching_objects_thin_as_if_alone(medial_surface):
    labels = _objects((50, 50, 50), 25, np.random.default_rng(4))
    together = _whole_volume(labels, medial_surface)
    alone = np.zeros_like(labels)
    for id_value in np.unique(labels[labels > 0]):
        thinned = _whole_volume((labels == id_value).astype(np.uint8), medial_surface)
        alone[thinned > 0] = id_value
    assert np.array_equal(together, alone)


def test_medial_surface_of_a_slab_is_one_voxel_thick():
    slab = np.zeros((30, 40, 40), np.uint8)
    slab[12:19, 5:35, 5:35] = 1
    surface = _whole_volume(slab, medial_surface=True)
    # away from its rim, one voxel per column, in the slab's middle
    core = surface[:, 12:28, 12:28]
    assert np.all(core.sum(axis=0) == 1)
    assert np.all(core[15] == 1)


@pytest.mark.parametrize("medial_surface", [False, True])
@pytest.mark.parametrize(
    "iterations_per_pass, block_shape",
    [(1, (52, 52, 52)), (1, (65, 52, 78)), (2, (104, 104, 104))],
)
def test_blockwise_matches_whole_volume(
    tmp_path, segmentation, medial_surface, iterations_per_pass, block_shape
):
    path, labels = segmentation
    thinning = TopologicalThinning(
        path,
        f"{tmp_path}/out.zarr/thinned",
        medial_surface=medial_surface,
        iterations_per_pass=iterations_per_pass,
        block_shape_voxels=block_shape,
        num_workers=1,
    )
    thinning.thin()
    result = thinning.output_vol.read_index((0, 0, 0), SHAPE)
    assert np.array_equal(result, _whole_volume(labels, medial_surface))


def test_roi_thins_like_a_crop_of_the_dataset(tmp_path, segmentation):
    path, labels = segmentation
    begin, end = np.array((7, 11, 3)), np.array((119, 118, 120))
    corner = np.array((16, 24, 40)) + begin * np.array(VOXEL_SIZE)
    roi = (corner, corner + (end - begin) * np.array(VOXEL_SIZE))
    thinning = TopologicalThinning(
        path,
        f"{tmp_path}/out.zarr/thinned",
        medial_surface=True,
        block_shape_voxels=(52, 52, 52),
        roi=roi,
        num_workers=1,
    )
    thinning.thin()
    result = thinning.output_vol.read_index((0, 0, 0), tuple(end - begin))
    crop = labels[tuple(slice(b, e) for b, e in zip(begin, end))]
    # the subfield order follows the dataset's own voxel index
    assert np.array_equal(result, _whole_volume(crop, True, origin=begin))


def test_mask_is_applied_before_thinning(tmp_path, segmentation):
    path, labels = segmentation
    mask = np.zeros(SHAPE, np.uint8)
    mask[:, :70, :] = 1
    mask_vol = create_multiscale_dataset(
        f"{tmp_path}/in.zarr/mask", dtype=np.uint8, grid=make_grid(SHAPE, VOXEL_SIZE, (16, 24, 40), chunks=(13, 13, 13))
    )
    mask_vol.write_index((0, 0, 0), SHAPE, mask)
    thinning = TopologicalThinning(
        path,
        f"{tmp_path}/out.zarr/thinned",
        block_shape_voxels=(52, 52, 52),
        mask_config={"cells": {"path": f"{tmp_path}/in.zarr/mask/s0", "mask_type": "inclusive"}},
        num_workers=1,
    )
    thinning.thin()
    result = thinning.output_vol.read_index((0, 0, 0), SHAPE)
    assert np.array_equal(result, _whole_volume(labels * mask, False))


def test_halo_must_fit_in_a_block(segmentation, tmp_path):
    path, _ = segmentation
    with pytest.raises(ValueError, match="halo"):
        TopologicalThinning(
            path,
            f"{tmp_path}/out.zarr/thinned",
            iterations_per_pass=2,
            block_shape_voxels=(52, 52, 52),
            num_workers=1,
        )


def test_deletions_reach_the_neighbours_that_need_them():
    deleted = np.zeros((10, 10, 10), bool)
    deleted[0, 5, 9] = True  # near the low z face and the high x face
    bands = deletion_bands(deleted, band=3)
    grid = np.zeros((3, 3, 3), np.int64)
    grid[1, 1, 1] = bands
    near = blocks_near_deletions(grid)
    expected = np.zeros((3, 3, 3), bool)
    for dz in (0, 1):  # the block itself and the one below it in z
        for dx in (1, 2):  # and the one after it in x
            expected[1 - dz, 1, dx] = True
    expected[1, 1, 1] = False
    assert np.array_equal(near, expected)

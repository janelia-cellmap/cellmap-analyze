import numpy as np
from cellmap_analyze.cythonizing.process_arrays import initialize_contact_site_array
from cellmap_analyze.cythonizing.bresenham3D import bresenham_3D_lines


# ---- initialize_contact_site_array tests ----


def _run_initialize(organelle_1, organelle_2, mask_out_surface_voxels=False):
    """Helper to call initialize_contact_site_array with pre-allocated output arrays."""
    surface_voxels_1 = np.zeros_like(organelle_1, dtype=np.uint8)
    surface_voxels_2 = np.zeros_like(organelle_2, dtype=np.uint8)
    mask = np.zeros_like(organelle_1, dtype=np.uint8)
    initialized_contact_sites = np.zeros_like(organelle_1, dtype=np.uint8)
    initialize_contact_site_array(
        organelle_1,
        organelle_2,
        surface_voxels_1,
        surface_voxels_2,
        mask,
        initialized_contact_sites,
        mask_out_surface_voxels,
    )
    return surface_voxels_1, surface_voxels_2, mask, initialized_contact_sites


def test_surface_detection():
    """A 3x3x3 cube in a 5x5x5 volume: all 26 outer voxels are surface, center is not."""
    org1 = np.zeros((5, 5, 5), dtype=np.uint8)
    org1[1:4, 1:4, 1:4] = 1

    # Second organelle: a 2x2x2 cube at a different location
    org2 = np.zeros((5, 5, 5), dtype=np.uint8)
    org2[1:3, 1:3, 1:3] = 1

    surface1, surface2, _, _ = _run_initialize(org1, org2)

    # organelle_1: interior voxel at center should NOT be surface
    expected_surface1 = np.zeros_like(org1, dtype=np.uint8)
    expected_surface1[1:4, 1:4, 1:4] = 1
    expected_surface1[2, 2, 2] = 0
    np.testing.assert_array_equal(surface1, expected_surface1)

    # organelle_2: 2x2x2 cube has no interior voxels, all should be surface
    expected_surface2 = np.zeros_like(org2, dtype=np.uint8)
    expected_surface2[1:3, 1:3, 1:3] = 1
    np.testing.assert_array_equal(surface2, expected_surface2)


def test_surface_detection_boundary_labels():
    """Surface detection correctly handles voxels at boundaries between different labels."""
    org1 = np.zeros((5, 5, 5), dtype=np.uint8)
    org1[1:4, 1:3, 1:4] = 1
    org1[1:4, 3:4, 1:4] = 2  # different label adjacent to label 1

    org2 = np.zeros((5, 5, 5), dtype=np.uint8)
    org2[1:3, 1:4, 1:3] = 3
    org2[1:3, 1:4, 3:4] = 4  # different label adjacent to label 3

    surface1, surface2, _, _ = _run_initialize(org1, org2)

    # organelle_1: voxels at the boundary between labels 1 and 2 (y=2 and y=3) are surface
    assert surface1[2, 2, 2] == 1  # label 1, neighbor at y=3 is label 2
    assert surface1[2, 3, 2] == 1  # label 2, neighbor at y=2 is label 1

    # organelle_2: voxels at the boundary between labels 3 and 4 (z=2 and z=3) are surface
    assert surface2[1, 2, 2] == 1  # label 3, neighbor at z=3 is label 4
    assert surface2[1, 2, 3] == 1  # label 4, neighbor at z=2 is label 3


def test_overlap_detection():
    """initialized_contact_sites is 1 exactly where both organelles are nonzero."""
    org1 = np.zeros((5, 5, 5), dtype=np.uint8)
    org2 = np.zeros((5, 5, 5), dtype=np.uint8)
    org1[1:4, 1:4, 1:3] = 1
    org2[1:4, 1:4, 2:4] = 1
    # Overlap is at z=2 (1:4, 1:4, 2:3)

    _, _, _, contact = _run_initialize(org1, org2)

    expected = np.zeros_like(org1, dtype=np.uint8)
    # range(1, nx-1) for nx=5 covers indices 1,2,3
    expected[1:4, 1:4, 2] = 1
    np.testing.assert_array_equal(contact, expected)


def test_mask_excludes_interior_only():
    """With mask_out_surface_voxels=False, mask marks interior (non-surface) organelle voxels."""
    org = np.zeros((5, 5, 5), dtype=np.uint8)
    org[1:4, 1:4, 1:4] = 1
    empty = np.zeros_like(org)

    surface, _, mask, _ = _run_initialize(org, empty, mask_out_surface_voxels=False)

    # Only the interior voxel (2,2,2) should be masked
    assert mask[2, 2, 2] == 1
    # Surface voxels should NOT be masked
    assert mask[1, 1, 1] == 0
    assert mask[1, 2, 2] == 0
    # Background should NOT be masked
    assert mask[0, 0, 0] == 0


def test_mask_excludes_all_organelle():
    """With mask_out_surface_voxels=True, mask marks all organelle voxels."""
    org = np.zeros((5, 5, 5), dtype=np.uint8)
    org[1:4, 1:4, 1:4] = 1
    empty = np.zeros_like(org)

    _, _, mask, _ = _run_initialize(org, empty, mask_out_surface_voxels=True)

    # All cube voxels (surface and interior) should be masked
    expected_mask = np.zeros_like(org, dtype=np.uint8)
    expected_mask[1:3, 1:3, 1:3] = 1  # only within range(1, nx-1)
    # Actually need to check what range(1, nx-1) covers for 5x5x5: x=1..3, y=1..3, z=1..3
    expected_mask = np.zeros_like(org, dtype=np.uint8)
    for x in range(1, 4):
        for y in range(1, 4):
            for z in range(1, 4):
                if org[x, y, z] > 0:
                    expected_mask[x, y, z] = 1
    np.testing.assert_array_equal(mask, expected_mask)


# ---- bresenham_3D_lines tests ----


def _run_bresenham(obj1_coords, obj2_coords, pairs, volume_shape, mask=None):
    """Helper to call bresenham_3D_lines with proper types."""
    obj1 = np.array(obj1_coords, dtype=np.int64)
    obj2 = np.array(obj2_coords, dtype=np.int64)
    contact_sites = np.zeros(volume_shape, dtype=np.uint8)
    if mask is None:
        mask = np.zeros(volume_shape, dtype=np.uint8)
    max_num_voxels = max(
        max(volume_shape) * 2, 10
    )  # generous buffer
    found = bresenham_3D_lines(
        pairs, obj1, obj2, contact_sites, max_num_voxels, mask
    )
    return found, contact_sites


def test_axis_aligned_line():
    """A line along the Z axis from (2,2,0) to (2,2,5) marks all voxels in between."""
    found, cs = _run_bresenham(
        obj1_coords=[[2, 2, 0]],
        obj2_coords=[[2, 2, 5]],
        pairs=[[0]],
        volume_shape=(5, 5, 8),
    )
    assert found is True
    for z in range(6):
        assert cs[2, 2, z] == 1, f"Expected voxel (2,2,{z}) to be marked"
    # Nothing else should be marked
    assert np.sum(cs) == 6


def test_diagonal_line():
    """A line from (0,0,0) to (4,4,4) marks the exact diagonal voxels."""
    found, cs = _run_bresenham(
        obj1_coords=[[0, 0, 0]],
        obj2_coords=[[4, 4, 4]],
        pairs=[[0]],
        volume_shape=(5, 5, 5),
    )
    assert found is True
    expected_voxels = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)]
    for voxel in expected_voxels:
        assert cs[voxel] == 1, f"Expected voxel {voxel} to be marked"
    assert np.sum(cs) == len(expected_voxels)


def test_mask_blocks_line():
    """When a mask blocks a voxel along the line, no voxels are marked."""
    mask = np.zeros((5, 5, 8), dtype=np.uint8)
    mask[2, 2, 3] = 1  # block the midpoint

    found, cs = _run_bresenham(
        obj1_coords=[[2, 2, 0]],
        obj2_coords=[[2, 2, 5]],
        pairs=[[0]],
        volume_shape=(5, 5, 8),
        mask=mask,
    )
    assert found is False
    assert np.sum(cs) == 0


def test_no_pairs_returns_false():
    """When there are no pairs to process, returns False with no voxels marked."""
    found, cs = _run_bresenham(
        obj1_coords=[[2, 2, 0]],
        obj2_coords=[[2, 2, 5]],
        pairs=[[]],  # voxel 0 of obj1 has no paired obj2 voxels
        volume_shape=(5, 5, 8),
    )
    assert found is False
    assert np.sum(cs) == 0


def test_multiple_pairs():
    """Two independent pairs produce two separate lines."""
    found, cs = _run_bresenham(
        obj1_coords=[[0, 2, 2], [2, 0, 2]],
        obj2_coords=[[4, 2, 2], [2, 4, 2]],
        pairs=[[0], [1]],  # obj1[0]→obj2[0], obj1[1]→obj2[1]
        volume_shape=(5, 5, 5),
    )
    assert found is True
    # Line 1: along X axis from (0,2,2) to (4,2,2)
    for x in range(5):
        assert cs[x, 2, 2] == 1, f"Expected voxel ({x},2,2) to be marked"
    # Line 2: along Y axis from (2,0,2) to (2,4,2)
    for y in range(5):
        assert cs[2, y, 2] == 1, f"Expected voxel (2,{y},2) to be marked"

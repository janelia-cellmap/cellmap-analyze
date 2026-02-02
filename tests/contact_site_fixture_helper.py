import numpy as np
from scipy.spatial import KDTree
from cellmap_analyze.cythonizing.bresenham3D import bresenham_3D_lines
import cc3d


def _find_surface_voxels(segmentation):
    """Find surface voxels using 6-connected neighbor check (pure NumPy).

    A voxel is a surface voxel if it is nonzero and at least one of its
    6 face-neighbors has a different value.
    """
    surface = np.zeros_like(segmentation, dtype=np.uint8)
    # Interior voxels only (skip boundary — matches Cython which iterates range(1, n-1))
    for axis in range(3):
        slc_center = [slice(1, -1)] * 3
        slc_minus = [slice(1, -1)] * 3
        slc_plus = [slice(1, -1)] * 3
        slc_minus[axis] = slice(0, -2)
        slc_plus[axis] = slice(2, None)
        center = segmentation[tuple(slc_center)]
        surface[tuple(slc_center)] |= (
            (center > 0)
            & (
                (center != segmentation[tuple(slc_minus)])
                | (center != segmentation[tuple(slc_plus)])
            )
        ).astype(np.uint8)
    return surface


def _build_mask(segmentation_1, segmentation_2, surface_voxels_1, surface_voxels_2):
    """Build the interior mask (non-surface organelle voxels that block Bresenham lines)."""
    mask = np.zeros_like(segmentation_1, dtype=np.uint8)
    # Interior of each organelle (voxels that are part of organelle but not surface)
    interior_1 = (segmentation_1 > 0) & (surface_voxels_1 == 0)
    interior_2 = (segmentation_2 > 0) & (surface_voxels_2 == 0)
    mask[interior_1 | interior_2] = 1
    return mask


def compute_contact_sites_ground_truth(
    segmentation_1, segmentation_2, contact_distance_nm, voxel_size
):
    """Compute contact site ground truth independently of ContactSites.get_ndarray_contact_sites.

    Uses pure NumPy for surface detection and KDTree pairing, reusing only
    the Bresenham line drawing (which operates in voxel space and is
    independently tested in test_contact_sites_subroutines.py).
    """
    voxel_size = np.asarray(voxel_size, dtype=float)
    contact_distance_voxels = contact_distance_nm / float(np.min(voxel_size))

    # Zero-pad to match get_ndarray_contact_sites(zero_pad=True)
    organelle_1 = np.pad(segmentation_1, 1)
    organelle_2 = np.pad(segmentation_2, 1)

    # Step 1: Find surface voxels (pure NumPy)
    surface_voxels_1 = _find_surface_voxels(organelle_1)
    surface_voxels_2 = _find_surface_voxels(organelle_2)

    # Mark overlapping voxels as contact sites directly
    current_pair_contact_sites = np.zeros_like(organelle_1, dtype=np.uint8)
    current_pair_contact_sites[(organelle_1 > 0) & (organelle_2 > 0)] = 1

    # Step 2: Build mask of interior voxels
    mask = _build_mask(organelle_1, organelle_2, surface_voxels_1, surface_voxels_2)

    # Step 3: KDTree pairing in physical space
    object_1_surface_voxel_coordinates = np.argwhere(surface_voxels_1)
    object_2_surface_voxel_coordinates = np.argwhere(surface_voxels_2)

    object_1_physical = object_1_surface_voxel_coordinates * voxel_size
    object_2_physical = object_2_surface_voxel_coordinates * voxel_size
    tree1 = KDTree(object_1_physical)
    tree2 = KDTree(object_2_physical)
    contact_voxels_list_of_lists = tree1.query_ball_tree(tree2, contact_distance_nm)

    # Step 4: Bresenham line drawing (voxel-space, independently tested)
    found_contact_voxels = bresenham_3D_lines(
        contact_voxels_list_of_lists,
        object_1_surface_voxel_coordinates,
        object_2_surface_voxel_coordinates,
        current_pair_contact_sites,
        2 * np.ceil(contact_distance_voxels),
        mask,
    )

    # Step 5: Connected components
    if found_contact_voxels:
        current_pair_contact_sites = cc3d.connected_components(
            current_pair_contact_sites,
            connectivity=26,
            binary_image=True,
        )

    # Remove padding
    return current_pair_contact_sites[1:-1, 1:-1, 1:-1].astype(np.uint64)

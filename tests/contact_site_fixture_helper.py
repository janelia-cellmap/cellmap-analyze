import numpy as np
from cellmap_analyze.process.contact_sites import ContactSites


def compute_contact_sites_ground_truth(
    segmentation_1, segmentation_2, contact_distance_nm, voxel_size
):
    """Compute contact site ground truth using the non-blockwise Bresenham-based method.

    Uses ContactSites.get_ndarray_contact_sites with voxel_size and
    contact_distance_nm to correctly handle anisotropic data.
    """
    voxel_size = np.asarray(voxel_size, dtype=float)
    contact_distance_voxels = contact_distance_nm / float(np.min(voxel_size))

    return ContactSites.get_ndarray_contact_sites(
        segmentation_1,
        segmentation_2,
        contact_distance_voxels,
        zero_pad=True,
        voxel_size=voxel_size,
        contact_distance_nm=contact_distance_nm,
    )

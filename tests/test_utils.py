import fastremap
import numpy as np
from cellmap_analyze.util.information_holders import ObjectInformation


def arrays_equal_up_to_id_ordering(arr1, arr2):
    test_uniques = fastremap.unique(arr1)
    gt_uniques = fastremap.unique(arr2)
    relabeling_dict = {}

    correct_uniques = True
    for test_unique in test_uniques:
        z, y, x = np.nonzero(arr1 == test_unique)
        cooresponding_gt_uniques = arr2[z, y, x]
        if len(fastremap.unique(cooresponding_gt_uniques)) > 1:
            correct_uniques = False
            break
        relabeling_dict[test_unique] = arr2[z[0], y[0], x[0]]

    fastremap.remap(arr1, relabeling_dict, preserve_missing_labels=True, in_place=True)
    return (
        correct_uniques
        and np.array_equal(arr1, arr2)
        and max(gt_uniques) == max(test_uniques)
    )


def radius_of_gyration_3d(segmentation, label=None, voxel_size=(1.0, 1.0, 1.0)):
    """
    Compute the radius of gyration for a 3D segmentation.

    Parameters
    ----------
    segmentation : ndarray, shape (Z, Y, X)
        3D numpy array of integer labels or binary mask.
    label : int or None
        If integer, compute Rg for that label (all voxels == label).
        If None, segmentation is treated as a binary mask (non-zero is foreground).
    voxel_size : tuple of float, length 3
        Physical size of a voxel in each dimension (dz, dy, dx).

    Returns
    -------
    Rg : float
        Radius of gyration in the same units as voxel_size.
    """
    # Create binary mask for the region of interest
    if label is None:
        mask = segmentation != 0
    else:
        mask = segmentation == label

    # Get voxel coordinates of the mask
    # coords will be an array of shape (N, 3): rows are (z, y, x)
    coords = np.array(np.nonzero(mask)).T
    if coords.size == 0:
        raise ValueError(f"No voxels found for label={label!r}")

    masses = np.ones(coords.shape[0], dtype=float)

    # Convert to physical coordinates
    voxel_size = np.asarray(voxel_size, dtype=float)
    phys_coords = coords * voxel_size

    # Compute center of mass
    total_mass = masses.sum()
    com = (phys_coords * masses[:, None]).sum(axis=0) / total_mass

    # Compute squared distances to the center of mass
    sq_dists = np.sum((phys_coords - com) ** 2, axis=1)

    # Radius of gyration is sqrt of mass-weighted mean squared distance
    Rg = np.sqrt((sq_dists * masses).sum() / total_mass)
    return Rg


def simple_surface_area(data, voxel_face_areas):
    """
    Calculate surface area with per-axis face areas for anisotropic voxels.

    Args:
        data: Binary mask array
        voxel_face_areas: Tuple of (Z-face area, Y-face area, X-face area) or scalar
    """
    # Handle both scalar and tuple voxel_face_areas
    if np.isscalar(voxel_face_areas):
        voxel_face_areas = (voxel_face_areas, voxel_face_areas, voxel_face_areas)

    data = np.pad(data, 1)
    s = data.shape
    sa = 0
    for i in range(1, s[0] - 1):
        for j in range(1, s[1] - 1):
            for k in range(1, s[2] - 1):
                if data[i, j, k]:
                    # Check Z-axis neighbors (perpendicular faces have area voxel_face_areas[0])
                    for delta in [-1, 1]:
                        if data[i + delta, j, k] == 0:
                            sa += voxel_face_areas[0]
                    # Check Y-axis neighbors (perpendicular faces have area voxel_face_areas[1])
                    for delta in [-1, 1]:
                        if data[i, j + delta, k] == 0:
                            sa += voxel_face_areas[1]
                    # Check X-axis neighbors (perpendicular faces have area voxel_face_areas[2])
                    for delta in [-1, 1]:
                        if data[i, j, k + delta] == 0:
                            sa += voxel_face_areas[2]
    return sa


def simple_contacting_surface_area(contact_site, segmentation, voxel_face_areas):
    """
    Calculate contacting surface area with per-axis face areas for anisotropic voxels.

    Args:
        contact_site: Binary mask of contact site
        segmentation: Binary mask of segmentation
        voxel_face_areas: Tuple of (Z-face area, Y-face area, X-face area) or scalar
    """
    # Handle both scalar and tuple voxel_face_areas
    if np.isscalar(voxel_face_areas):
        voxel_face_areas = (voxel_face_areas, voxel_face_areas, voxel_face_areas)

    segmentation = np.pad(segmentation, 1)
    contact_site = np.pad(contact_site, 1)
    s = contact_site.shape
    sa = 0
    for i in range(1, s[0] - 1):
        for j in range(1, s[1] - 1):
            for k in range(1, s[2] - 1):
                if contact_site[i, j, k] and segmentation[i, j, k]:
                    # Check Z-axis neighbors (perpendicular faces have area voxel_face_areas[0])
                    for delta in [-1, 1]:
                        if segmentation[i + delta, j, k] == 0:
                            sa += voxel_face_areas[0]
                    # Check Y-axis neighbors (perpendicular faces have area voxel_face_areas[1])
                    for delta in [-1, 1]:
                        if segmentation[i, j + delta, k] == 0:
                            sa += voxel_face_areas[1]
                    # Check X-axis neighbors (perpendicular faces have area voxel_face_areas[2])
                    for delta in [-1, 1]:
                        if segmentation[i, j, k + delta] == 0:
                            sa += voxel_face_areas[2]
    return sa


def simple_bounding_box(data, voxel_size):
    z, y, x = np.where(data)

    # add 0.5 to center on voxel
    z = z + 0.5
    y = y + 0.5
    x = x + 0.5

    # Handle both scalar and tuple voxel_size
    if np.isscalar(voxel_size):
        voxel_size = (voxel_size, voxel_size, voxel_size)

    z *= voxel_size[0]
    y *= voxel_size[1]
    x *= voxel_size[2]
    return [min(z), min(y), min(x), max(z), max(y), max(x)]


def simple_com(data, voxel_size):
    idxs = np.where(data)
    # Handle both scalar and tuple voxel_size
    if np.isscalar(voxel_size):
        voxel_size = np.array([voxel_size, voxel_size, voxel_size])
    else:
        voxel_size = np.array(voxel_size)
    com = (np.mean(idxs, axis=1) + 0.5) * voxel_size
    return com


def simple_id_to_surface_area_dict(contact_site, segmentation, voxel_face_areas):
    id_to_surface_area_dict = {}
    for id in np.unique(segmentation[segmentation > 0]):
        sa = simple_contacting_surface_area(
            contact_site, segmentation == id, voxel_face_areas
        )
        if sa > 0:
            id_to_surface_area_dict[id] = sa
    return id_to_surface_area_dict


def simple_sum_r2(data, voxel_size):
    idxs = np.where(data)
    positions = np.array(idxs).T + 0.5  # center on voxel
    # Handle both scalar and tuple voxel_size
    if np.isscalar(voxel_size):
        voxel_size = np.array([voxel_size, voxel_size, voxel_size])
    else:
        voxel_size = np.array(voxel_size)
    positions *= voxel_size
    return np.sum(np.sum(positions**2, axis=1))


def simple_object_information_dict(segmentation, voxel_size):
    # Handle both scalar and tuple voxel_size
    if np.isscalar(voxel_size):
        voxel_size = np.array([voxel_size, voxel_size, voxel_size])
    else:
        voxel_size = np.array(voxel_size)

    voxel_volume = np.prod(voxel_size)
    # Calculate per-axis face areas (perpendicular to each axis)
    voxel_face_areas = (
        voxel_size[1] * voxel_size[2],  # Z-axis faces: Y × X
        voxel_size[0] * voxel_size[2],  # Y-axis faces: Z × X
        voxel_size[0] * voxel_size[1],  # X-axis faces: Z × Y
    )
    object_information_dict = {}
    for id in np.unique(segmentation[segmentation > 0]):
        obj = segmentation == id
        # get positions of obj
        counts = np.sum(obj)
        object_information_dict[id] = ObjectInformation(
            counts=counts,
            volume=counts * voxel_volume,
            com=simple_com(obj, voxel_size),
            surface_area=simple_surface_area(obj, voxel_face_areas),
            sum_r2=simple_sum_r2(obj, voxel_size),
            bounding_box=simple_bounding_box(obj, voxel_size),
        )
        assert np.allclose(
            object_information_dict[id].radius_of_gyration,
            radius_of_gyration_3d(segmentation, label=id, voxel_size=tuple(voxel_size)),
        )

    return object_information_dict


def simple_contact_site_information_dict(
    segmentation_1, segmentation_2, contact_site, voxel_size
):
    # Handle both scalar and tuple voxel_size
    if np.isscalar(voxel_size):
        voxel_size = np.array([voxel_size, voxel_size, voxel_size])
    else:
        voxel_size = np.array(voxel_size)

    voxel_volume = np.prod(voxel_size)
    # Calculate per-axis face areas (perpendicular to each axis)
    voxel_face_areas = (
        voxel_size[1] * voxel_size[2],  # Z-axis faces: Y × X
        voxel_size[0] * voxel_size[2],  # Y-axis faces: Z × X
        voxel_size[0] * voxel_size[1],  # X-axis faces: Z × Y
    )
    contact_site_information_dict = {}
    for id in np.unique(contact_site[contact_site > 0]):
        cs = contact_site == id
        contact_site_information_dict[id] = ObjectInformation(
            counts=np.sum(cs),
            volume=np.sum(cs) * voxel_volume,
            com=simple_com(cs, voxel_size),
            sum_r2=simple_sum_r2(cs, voxel_size),
            surface_area=simple_surface_area(cs, voxel_face_areas),
            bounding_box=simple_bounding_box(cs, voxel_size),
            id_to_surface_area_dict_1=simple_id_to_surface_area_dict(
                cs, segmentation_1, voxel_face_areas
            ),
            id_to_surface_area_dict_2=simple_id_to_surface_area_dict(
                cs, segmentation_2, voxel_face_areas
            ),
        )
    return contact_site_information_dict


def simple_raw_intensity_stats(segmentation, raw_data):
    """Compute per-label intensity stats as ground truth."""
    stats = {}
    for label_id in np.unique(segmentation[segmentation > 0]):
        values = raw_data[segmentation == label_id].astype(np.float64)
        stats[label_id] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "sum": np.sum(values),
            "sum_sq": np.sum(values**2),
            "count": len(values),
        }
    return stats

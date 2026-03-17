"""Fixtures for tests/operations/ — data generation, zarr writing, and CSV setup."""

import pytest
import numpy as np
from funlib.geometry import Roi
from skimage import measure
from cellmap_analyze.util.zarr_util import create_multiscale_dataset
import os
from scipy import ndimage
import pandas as pd
from tests.test_utils import simple_object_information_dict
from funlib.persistence import prepare_ds
from tests.contact_site_fixture_helper import compute_contact_sites_ground_truth


# =============================================================================
# Segmentation data fixtures
# =============================================================================


@pytest.fixture(scope="session")
def segmentation_1(image_shape):
    seg = np.zeros(image_shape, dtype=np.uint8)
    seg[:2, 2:10, 2:10] = 1
    return seg


@pytest.fixture(scope="session")
def segmentation_1_downsampled(segmentation_1):
    # take every other pixel
    return segmentation_1[::2, ::2, ::2]


@pytest.fixture(scope="session")
def segmentation_2(image_shape):
    seg = np.zeros(image_shape, dtype=np.uint8)
    seg[3:5, 2:5, 4:6] = 1
    seg[3:9, 7:10, 4:8] = 2
    return seg


@pytest.fixture(scope="session")
def segmentation_random(image_shape):
    np.random.seed(42)
    seg = np.random.randint(low=1, high=10, size=image_shape, dtype=np.uint8)
    return seg


@pytest.fixture(scope="session")
def raw_intensity_for_seg2(image_shape):
    np.random.seed(123)
    return np.random.rand(*image_shape).astype(np.float32) * 255


@pytest.fixture(autouse=True, scope="session")
def segmentation_cells(image_shape):
    seg = np.zeros(image_shape, dtype=np.uint8)
    # assign different values to different corners
    seg[:2, :2, :2] = 1
    seg[-2:, -2:, -2:] = 2
    seg[5:8, 5:8, 5:8] = 3
    return seg


# =============================================================================
# Connected components / fill holes / morphology fixtures
# =============================================================================


@pytest.fixture(scope="session")
def image_with_holes(image_shape):
    seg = np.zeros(image_shape, dtype=np.uint8)

    # corner cube
    seg[:3, :3, :3] = 1
    seg[0, 0, 0] = 0  # shouldnt be filled since it is on border
    seg[1, 1, 1] = 0  # should be filled

    # hole spanning chunk
    seg[3:7, 3:7, 3:7] = 2
    seg[4:6, 4:6, 4:6] = 0

    # two rectangles on top of eachother
    seg[7:11, 7:11, 7:9] = 3
    seg[7:11, 7:11, 9:11] = 4
    seg[8:10, 8:10, 8:10] = 0  # hole between objects should be kept as a hole

    return seg


@pytest.fixture(scope="session")
def image_with_holes_filled(image_with_holes):
    filled = np.zeros_like(image_with_holes)
    for id in np.unique(image_with_holes[image_with_holes > 0]):
        filled += (
            ndimage.binary_fill_holes(image_with_holes == id).astype(np.uint8)
        ) * id
    return filled


@pytest.fixture(scope="session")
def random_image_to_delete(image_shape):
    np.random.seed(42)
    return np.random.randint(low=0, high=255, size=image_shape, dtype=np.uint8)


@pytest.fixture(scope="session")
def tic_tac_toe(image_shape):
    seg = np.zeros(image_shape, dtype=np.uint16)
    id = 2**8 + 1000
    seg[3, :, :] = id
    seg[7, :, :] = id
    seg[:, 3, :] = id
    seg[:, 7, :] = id
    seg[:, :, 3] = id
    seg[:, :, 7] = id
    return seg


@pytest.fixture(scope="session")
def tic_tac_toe_filled(tic_tac_toe):
    seg = ndimage.binary_fill_holes(tic_tac_toe).astype(np.uint16)
    seg[seg > 0] = 2**8 + 1000
    return seg


@pytest.fixture(scope="session")
def duplicate_ids(image_shape):
    seg = np.zeros(image_shape, dtype=np.uint8)
    seg[:7, :7, :7] = 1
    seg[:7, 7:9, 7:9] = 2
    seg[:7, 7:11, 9:11] = 1
    return seg


@pytest.fixture(scope="session")
def no_duplicate_ids(image_shape):
    seg = np.zeros(image_shape, dtype=np.uint8)
    seg[:7, :7, :7] = 1
    seg[:7, 7:9, 7:9] = 2
    seg[:7, 7:11, 9:11] = 3
    return seg


@pytest.fixture(scope="session")
def binarizable_image(image_shape):
    seg = np.zeros(image_shape, dtype=np.uint8)
    seg[0, 0, 0] = 1
    seg[0, 1, 0] = 2
    seg[0, 1:, 1] = 1
    return seg


@pytest.fixture(scope="session")
def intensity_image(image_shape):
    seg = np.zeros(image_shape, dtype=np.uint8)
    seg[1, 1, 1] = 127
    for x in range(3, 5):
        for y in range(3, 5):
            for z in range(3, 5):
                seg[x, y, z] = 127
    seg[7:11, 7:11, 7:11] = 127
    return seg


@pytest.fixture(scope="session")
def connected_components(intensity_image):
    seg = measure.label(intensity_image > 0, connectivity=1)
    return seg


@pytest.fixture(scope="session")
def blockwise_connected_components(image_shape):
    seg = np.zeros(image_shape, dtype=np.uint64)
    seg[1, 1, 1] = 1
    id = 2
    for x in range(3, 5):
        for y in range(3, 5):
            for z in range(3, 5):
                seg[x, y, z] = id
                id += 1
    seg[7:11, 7:11, 7:11] = id + 1
    return seg


# =============================================================================
# Mask fixtures
# =============================================================================


@pytest.fixture(scope="session")
def mask_one(image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[:3, :3, :3] = 1
    mask[4:6, 2:3, 4:6] = 2
    return mask


@pytest.fixture(scope="session")
def mask_two(image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[7:11, 7:11, 7:9] = 1
    mask[7:11, 7:11, 9:11] = 2
    return mask


@pytest.fixture(scope="session")
def label_mask(image_shape):
    mask = np.zeros(image_shape, dtype=np.uint64)
    mask[:5, :5, :5] = 1_000_000_000
    mask[6:, 6:, :] = 2_000_000_0000
    return mask


# =============================================================================
# Cylinder / sphere / affinity fixtures
# =============================================================================


@pytest.fixture(scope="session")
def horizontal_cylinder_endpoints(voxel_size):
    coords = np.array([[46, 3, 3], [46, 3, 40]], dtype=float)
    return coords * voxel_size + 0.5 * voxel_size


@pytest.fixture(scope="session")
def vertical_cylinder_endpoints(voxel_size):
    coords = np.array([[2, 2, 2], [2, 48, 2]], dtype=float)
    return coords * voxel_size + 0.5 * voxel_size


@pytest.fixture(scope="session")
def diagonal_cylinder_endpoints(voxel_size):
    coords = np.array([[45, 45, 5], [5, 5, 45]], dtype=float)
    return coords * voxel_size + 0.5 * voxel_size


@pytest.fixture(scope="session")
def segmentation_cylinders(
    horizontal_cylinder_endpoints,
    vertical_cylinder_endpoints,
    diagonal_cylinder_endpoints,
    voxel_size,
):
    def fill_in_cylinder(seg, endpoints_voxel, radius_nm, id, voxel_size):
        end_2_voxel = endpoints_voxel[0] - 0.5
        end_1_voxel = endpoints_voxel[1] - 0.5

        end_2 = end_2_voxel * voxel_size
        end_1 = end_1_voxel * voxel_size

        d = np.divide(end_2 - end_1, np.linalg.norm(end_2 - end_1))

        radius_voxels = np.ceil(radius_nm / voxel_size).astype(int)
        mins = np.floor(np.minimum(end_1_voxel, end_2_voxel)).astype(int) - (
            radius_voxels + 1
        )
        maxs = np.ceil(np.maximum(end_1_voxel, end_2_voxel)).astype(int) + (
            radius_voxels + 1
        )

        z, y, x = [list(range(mins[i], maxs[i] + 1, 1)) for i in range(3)]
        p_voxel = np.array(np.meshgrid(z, y, x)).T.reshape((-1, 3))

        p = p_voxel * voxel_size

        s = np.dot(end_1 - p, d)
        t = np.dot(p - end_2, d)

        h = np.maximum.reduce([s, t, np.zeros_like(s)])

        c = np.linalg.norm(np.cross(p - end_1, d), axis=1)

        is_in_cylinder = (h == 0) & (c <= radius_nm)
        voxels_in_cylinder = p_voxel[is_in_cylinder]
        seg[
            voxels_in_cylinder[:, 0], voxels_in_cylinder[:, 1], voxels_in_cylinder[:, 2]
        ] = id

    seg = np.zeros((50, 50, 50), dtype=np.uint8)
    min_voxel = np.min(voxel_size)
    fill_in_cylinder(seg, horizontal_cylinder_endpoints / voxel_size, 1 * min_voxel, 1, voxel_size)
    seg[0, 0, 0] = 1  # add spurious voxel
    fill_in_cylinder(seg, vertical_cylinder_endpoints / voxel_size, 1.5 * min_voxel, 2, voxel_size)
    fill_in_cylinder(seg, diagonal_cylinder_endpoints / voxel_size, 2 * min_voxel, 3, voxel_size)

    return seg


@pytest.fixture(scope="session")
def segmentation_spheres(
    shape=(50, 50, 50),
    spheres=[((12, 12, 12), 5), ((25, 25, 25), 20), ((40, 40, 40), 8)],
):
    seg = np.zeros(shape, dtype=np.uint8)
    z, y, x = np.indices(shape)
    for label, ((zc, yc, xc), r) in enumerate(spheres, start=1):
        mask = (z - zc) ** 2 + (y - yc) ** 2 + (x - xc) ** 2 < r**2
        seg[mask] = label
    seg[0:6, 15:36, 22:29] = 2
    return seg


@pytest.fixture(scope="session")
def affinities_offsets():
    return [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
        [9, 0, 0],
        [0, 9, 0],
        [0, 0, 9],
    ]


@pytest.fixture(scope="session")
def affinities_cylinders(segmentation_cylinders, affinities_offsets):
    shape = segmentation_cylinders.shape
    num_offsets = len(affinities_offsets)
    out = np.zeros((num_offsets,) + shape, dtype=np.uint8)

    for idx, (dx, dy, dz) in enumerate(affinities_offsets):
        shifted = np.zeros(shape, dtype=segmentation_cylinders.dtype)

        src_slice = [
            slice(max(dx, 0), shape[0] + min(dx, 0)),
            slice(max(dy, 0), shape[1] + min(dy, 0)),
            slice(max(dz, 0), shape[2] + min(dz, 0)),
        ]

        dst_slice = [
            slice(max(-dx, 0), shape[0] + min(-dx, 0)),
            slice(max(-dy, 0), shape[1] + min(-dy, 0)),
            slice(max(-dz, 0), shape[2] + min(-dz, 0)),
        ]

        shifted[tuple(dst_slice)] = segmentation_cylinders[tuple(src_slice)]

        out[idx] = (
            (segmentation_cylinders == shifted).astype(np.uint8)
            * (segmentation_cylinders > 0)
            * 255
        )
    return out


# =============================================================================
# Contact site ground truth fixtures
# =============================================================================


@pytest.fixture(scope="session")
def contact_sites_distance_8nm(segmentation_1, segmentation_2, voxel_size):
    """Ground truth contact sites for contact_distance_nm=8."""
    return compute_contact_sites_ground_truth(
        segmentation_1, segmentation_2, contact_distance_nm=8, voxel_size=voxel_size
    )


@pytest.fixture(scope="session")
def contact_sites_distance_16nm(segmentation_1, segmentation_2, voxel_size):
    """Ground truth contact sites for contact_distance_nm=16."""
    return compute_contact_sites_ground_truth(
        segmentation_1, segmentation_2, contact_distance_nm=16, voxel_size=voxel_size
    )


@pytest.fixture(scope="session")
def contact_sites_distance_24nm(segmentation_1, segmentation_2, voxel_size):
    """Ground truth contact sites for contact_distance_nm=24."""
    return compute_contact_sites_ground_truth(
        segmentation_1, segmentation_2, contact_distance_nm=24, voxel_size=voxel_size
    )


# =============================================================================
# Skeleton fixture
# =============================================================================


@pytest.fixture(scope="session")
def segmentation_for_skeleton():
    """Create a segmentation with multiple distinct IDs of various shapes for skeletonization tests."""
    seg = np.zeros((50, 50, 50), dtype=np.uint8)

    # ID 1: Small cube in corner
    seg[1:4, 1:4, 1:4] = 1

    # ID 2: Elongated structure (good for skeletonization)
    seg[5:15, 2:5, 2:5] = 2

    # ID 3: Another elongated structure (perpendicular to ID 2)
    seg[2:5, 5:15, 2:5] = 3

    # ID 4: Sphere (should skeletonize to a point or small cluster)
    center = (25, 25, 25)
    radius = 4
    z, y, x = np.ogrid[0:50, 0:50, 0:50]
    sphere_mask = (z - center[0]) ** 2 + (y - center[1]) ** 2 + (
        x - center[2]
    ) ** 2 <= radius**2
    seg[sphere_mask] = 4

    # ID 5: Cross/plus shape (3D cross with branches in Z, X, Y directions)
    # Thick arms (4x4 cross-section) so skeletonize produces real branches.
    seg[30:50, 23:27, 23:27] = 5
    seg[38:42, 23:27, 10:40] = 5
    seg[38:42, 14:36, 23:27] = 5

    # ID 6: L-shaped structure
    seg[5:10, 30:35, 30:32] = 6
    seg[8:10, 30:32, 30:40] = 6

    # ID 7: Figure-8 shape (two loops in a single z-plane)
    z, y, x = np.ogrid[0:50, 0:50, 0:50]

    z_plane = 15
    plane_mask = z == z_plane

    bottom_center = (45, 15)
    top_center = (45, 23)
    radius = 3

    bottom_ring = (
        plane_mask
        & (
            (y - bottom_center[0]) ** 2 + (x - bottom_center[1]) ** 2
            >= (radius - 0.5) ** 2
        )
        & (
            (y - bottom_center[0]) ** 2 + (x - bottom_center[1]) ** 2
            <= (radius + 0.5) ** 2
        )
    )
    seg[bottom_ring] = 7

    top_ring = (
        plane_mask
        & ((y - top_center[0]) ** 2 + (x - top_center[1]) ** 2 >= (radius - 0.5) ** 2)
        & ((y - top_center[0]) ** 2 + (x - top_center[1]) ** 2 <= (radius + 0.5) ** 2)
    )
    seg[top_ring] = 7

    seg[z_plane, 44:47, 18:20] = 7

    return seg


# =============================================================================
# Zarr writing and CSV setup
# =============================================================================


@pytest.fixture(scope="session")
def tmp_zarr(shared_tmpdir, voxel_size):
    # Create separate zarr files for each voxel_size parametrization
    voxel_size_str = f"{voxel_size[0]}_{voxel_size[1]}_{voxel_size[2]}"
    output_path = shared_tmpdir + f"/tmp_{voxel_size_str}.zarr"
    os.makedirs(name=output_path, exist_ok=True)
    return str(output_path)


@pytest.fixture(scope="session")
def test_image_dict(
    blockwise_connected_components,
    connected_components,
    intensity_image,
    image_with_holes,
    image_with_holes_filled,
    tic_tac_toe,
    tic_tac_toe_filled,
    no_duplicate_ids,
    duplicate_ids,
    binarizable_image,
    mask_one,
    mask_two,
    label_mask,
    random_image_to_delete,
    segmentation_1,
    segmentation_2,
    segmentation_random,
    segmentation_cylinders,
    segmentation_spheres,
    segmentation_cells,
    affinities_cylinders,
    segmentation_1_downsampled,
    contact_sites_distance_8nm,
    contact_sites_distance_16nm,
    contact_sites_distance_24nm,
    segmentation_for_skeleton,
    raw_intensity_for_seg2,
):
    return {
        "blockwise_connected_components": blockwise_connected_components,
        "connected_components": connected_components,
        "intensity_image": intensity_image,
        "image_with_holes": image_with_holes,
        "image_with_holes_filled": image_with_holes_filled,
        "tic_tac_toe": tic_tac_toe,
        "tic_tac_toe_filled": tic_tac_toe_filled,
        "binarizable_image": binarizable_image,
        "duplicate_ids": duplicate_ids,
        "no_duplicate_ids": no_duplicate_ids,
        "mask_one": mask_one,
        "mask_two": mask_two,
        "label_mask": label_mask,
        "random_image_to_delete": random_image_to_delete,
        "segmentation_1": segmentation_1,
        "segmentation_cylinders": segmentation_cylinders,
        "segmentation_spheres": segmentation_spheres,
        "affinities_cylinders": affinities_cylinders,
        "segmentation_1_downsampled": segmentation_1_downsampled,
        "segmentation_2": segmentation_2,
        "segmentation_random": segmentation_random,
        "segmentation_cells": segmentation_cells,
        "segmentation_for_skeleton": segmentation_for_skeleton,
        "raw_intensity_for_seg2": raw_intensity_for_seg2,
        "contact_sites_distance_8nm": contact_sites_distance_8nm,
        "contact_sites_distance_16nm": contact_sites_distance_16nm,
        "contact_sites_distance_24nm": contact_sites_distance_24nm,
    }


@pytest.fixture(autouse=True, scope="session")
def write_zarrs(tmp_zarr, test_image_dict, voxel_size, chunk_size):
    for data_name, data in test_image_dict.items():
        current_voxel_size = (
            voxel_size if "downsampled" not in data_name else 2 * voxel_size
        )
        data_path = f"{tmp_zarr}/{data_name}"

        ds_voxel_size = tuple(int(v) for v in current_voxel_size)
        data_shape = data.shape
        if len(data_shape) == 4:
            data_shape = data_shape[1:]

        total_roi = Roi((0, 0, 0), np.array(data_shape) * current_voxel_size)
        write_size = (
            chunk_size * current_voxel_size
            if (
                data_name != "segmentation_cylinders"
                and data_name != "affinities_cylinders"
                and data_name != "segmentation_spheres"
                and data_name != "segmentation_for_skeleton"
            )
            else np.array((20, 20, 20)) * current_voxel_size
        )

        if "affinities" in data_name:
            ds = prepare_ds(
                tmp_zarr,
                data_name,
                total_roi=total_roi,
                write_size=write_size,
                voxel_size=ds_voxel_size,
                dtype=np.uint8,
                num_channels=np.shape(data)[0],
            )
        else:
            ds = create_multiscale_dataset(
                data_path,
                dtype=data.dtype,
                voxel_size=ds_voxel_size,
                total_roi=total_roi,
                write_size=write_size,
            )

        ds.data[:] = data


@pytest.fixture(autouse=True, scope="session")
def tmp_object_information_csv(shared_tmpdir):
    output_path = shared_tmpdir + "/csvs/"
    os.makedirs(name=output_path, exist_ok=True)
    df = pd.DataFrame(
        {
            "Object ID": [1, 4],
            "Volume (nm^3)": [100, 400],
            "Surface Area (nm^2)": [100, 400],
        }
    )
    df.to_csv(output_path + "/objects.csv", index=False)
    return str(output_path + "/objects.csv")


@pytest.fixture(autouse=True, scope="session")
def tmp_coms_csv(shared_tmpdir, voxel_size):
    output_path = shared_tmpdir + "/csvs/"
    os.makedirs(name=output_path, exist_ok=True)
    df = pd.DataFrame(
        {
            "Object ID": [1, 2, 3],
            "COM X (nm)": (np.array([1.1, 3.1, 11.1]) + 0.5) * voxel_size[2],
            "COM Y (nm)": (np.array([1.1, 3.1, 11.1]) + 0.5) * voxel_size[1],
            "COM Z (nm)": (np.array([1.1, 3.1, 8.1]) + 0.5) * voxel_size[0],
        }
    )
    df.to_csv(output_path + "/assignment_coms.csv", index=False)
    return str(output_path + "/assignment_coms.csv")


@pytest.fixture(autouse=True, scope="session")
def tmp_cylinders_information_csv(segmentation_cylinders, shared_tmpdir, voxel_size):
    output_path = shared_tmpdir + "/csvs/"
    os.makedirs(name=output_path, exist_ok=True)
    cylinder_information_dict = simple_object_information_dict(
        segmentation_cylinders, voxel_size
    )
    columns = [
        "Object ID",
        "Volume (nm^3)",
        "Surface Area (nm^2)",
        "Radius of Gyration (nm)",
    ]
    for category in ["COM", "MIN", "MAX"]:
        for d in ["X", "Y", "Z"]:
            columns.append(f"{category} {d} (nm)")
    df = pd.DataFrame(
        index=np.arange(len(cylinder_information_dict)),
        columns=columns,
    )
    for i, (id, oi) in enumerate(cylinder_information_dict.items()):
        row = [
            id,
            float(oi.volume),
            float(oi.surface_area),
            oi.radius_of_gyration,
            *oi.com[::-1],
            *oi.bounding_box[:3][::-1],
            *oi.bounding_box[3:][::-1],
        ]
        df.loc[i] = row

    df.to_csv(output_path + "/segmentation_cylinders.csv", index=False)
    return str(output_path + "/segmentation_cylinders.csv")

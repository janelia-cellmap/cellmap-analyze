import pytest
import numpy as np
from funlib.geometry import Roi
from skimage import measure
from cellmap_analyze.util.zarr_util import create_multiscale_dataset
import os
from scipy import ndimage
import pandas as pd
from tests.operations.test_measure import simple_object_information_dict
from funlib.persistence import prepare_ds


@pytest.fixture(autouse=True, scope="session")
def voxel_size():
    return 8


@pytest.fixture(scope="session")
def horizontal_cylinder_endpoints(voxel_size):
    return np.array([[46, 3, 3], [46, 3, 40]]) * 8 + 0.5 * voxel_size


@pytest.fixture(scope="session")
def vertical_cylinder_endpoints(voxel_size):
    return np.array([[2, 2, 2], [2, 48, 2]]) * voxel_size + 0.5 * voxel_size


@pytest.fixture(scope="session")
def diagonal_cylinder_endpoints(voxel_size):
    return np.array([[45, 45, 5], [5, 5, 45]]) * voxel_size + 0.5 * voxel_size


@pytest.fixture(scope="session")
def image_shape():
    return np.array((11, 11, 11))


@pytest.fixture(scope="session")
def chunk_size():
    return np.array((4, 4, 4))


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
def random_image_to_delete(image_shape):
    np.random.seed(42)
    return np.random.randint(low=0, high=255, size=image_shape, dtype=np.uint8)


@pytest.fixture(scope="session")
def tic_tac_toe(image_shape):
    seg = np.zeros(image_shape, dtype=np.uint16)
    id = 2**8 + 1000
    # Set the two slices along axis-0 (i.e., seg[3, :, :] and seg[7, :, :]) to 1
    seg[3, :, :] = id
    seg[7, :, :] = id

    # Set the two slices along axis-1 (i.e., A[:, 3, :] and A[:, 7, :]) to 1
    seg[:, 3, :] = id
    seg[:, 7, :] = id

    # Set the two slices along axis-2 (i.e., seg[:, :, 3] and seg[:, :, 7]) to 1
    seg[:, :, 3] = id
    seg[:, :, 7] = id

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


@pytest.fixture(scope="session")
def image_with_holes_filled(image_with_holes):
    filled = np.zeros_like(image_with_holes)
    for id in np.unique(image_with_holes[image_with_holes > 0]):
        filled += (
            ndimage.binary_fill_holes(image_with_holes == id).astype(np.uint8)
        ) * id
    return filled


@pytest.fixture(scope="session")
def tic_tac_toe_filled(tic_tac_toe):
    seg = ndimage.binary_fill_holes(tic_tac_toe).astype(np.uint16)
    seg[seg > 0] = 2**8 + 1000
    return seg


@pytest.fixture(scope="session")
def blockwise_connected_components(image_shape):
    seg = np.zeros(image_shape, dtype=np.uint64)
    # single voxel
    seg[1, 1, 1] = 1

    # cube around border
    id = 2
    for x in range(3, 5):
        for y in range(3, 5):
            for z in range(3, 5):
                seg[x, y, z] = id
                id += 1

    seg[7:11, 7:11, 7:11] = id + 1
    return seg


@pytest.fixture(scope="session")
def intensity_image(image_shape):
    seg = np.zeros(image_shape, dtype=np.uint8)
    # single voxel
    seg[1, 1, 1] = 127

    # cube around border
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
def segmentation_cylinders(
    horizontal_cylinder_endpoints,
    vertical_cylinder_endpoints,
    diagonal_cylinder_endpoints,
    voxel_size,
):
    def fill_in_cylinder(seg, endpoints, radius, id):
        # subtract 0.5 so that an annotation centered at a voxel matches to eg 0,0,0
        end_2 = endpoints[0] - 0.5
        end_1 = endpoints[1] - 0.5
        # https://stackoverflow.com/questions/56463412/distance-from-a-point-to-a-line-segment-in-3d-python
        # normalized tangent vector
        d = np.divide(end_2 - end_1, np.linalg.norm(end_2 - end_1))

        # possible points
        mins = np.floor(np.minimum(end_1, end_2)).astype(int) - (
            np.ceil(radius).astype(int) + 1
        )  # 1s for padding
        maxs = np.ceil(np.maximum(end_1, end_2)).astype(int) + (
            np.ceil(radius).astype(int) + 1
        )

        z, y, x = [list(range(mins[i], maxs[i] + 1, 1)) for i in range(3)]
        p = np.array(np.meshgrid(z, y, x)).T.reshape((-1, 3))

        # signed parallel distance components
        s = np.dot(end_1 - p, d)
        t = np.dot(p - end_2, d)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, np.zeros_like(s)])

        # perpendicular distance component
        c = np.linalg.norm(np.cross(p - end_1, d), axis=1)

        is_in_cylinder = (h == 0) & (c <= radius)
        voxels_in_cylinder = p[is_in_cylinder]
        seg[
            voxels_in_cylinder[:, 0], voxels_in_cylinder[:, 1], voxels_in_cylinder[:, 2]
        ] = id

    # need to use larger to get better chance of fitting
    seg = np.zeros((50, 50, 50), dtype=np.uint8)
    # horizontal cylinder
    fill_in_cylinder(seg, horizontal_cylinder_endpoints / voxel_size, 1, 1)
    seg[0, 0, 0] = 1  # add spurious voxel
    # vertical cylinder
    fill_in_cylinder(seg, vertical_cylinder_endpoints / voxel_size, 1.5, 2)
    # diagonal
    fill_in_cylinder(seg, diagonal_cylinder_endpoints / voxel_size, 2, 3)

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
    """
    Create a new array with shape (num_offsets, *seg.shape) where each channel
    is 1 if the pixel at the offset from the current pixel belongs to the same object,
    and 0 otherwise.

    Parameters:
      seg: numpy array of shape (D, H, W) containing segmentation labels.
      offsets: list of tuples, e.g. [(1,0,0), (0,1,0), ...] representing the offsets.

    Returns:
      out: numpy array of shape (len(offsets), D, H, W)
    """
    shape = segmentation_cylinders.shape
    num_offsets = len(affinities_offsets)
    out = np.zeros((num_offsets,) + shape, dtype=np.uint8)

    for idx, (dx, dy, dz) in enumerate(affinities_offsets):
        # Create an array for the shifted segmentation; fill with -1 to mark invalid areas.
        shifted = np.zeros(shape, dtype=segmentation_cylinders.dtype)

        # Compute slices for the valid region in the source and destination.
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

        # Shift the segmentation array.
        shifted[tuple(dst_slice)] = segmentation_cylinders[tuple(src_slice)]

        # For each pixel, assign 1 if the label matches that of the shifted pixel.
        out[idx] = (
            (segmentation_cylinders == shifted).astype(np.uint8)
            * (segmentation_cylinders > 0)
            * 255
        )
    return out


@pytest.fixture(autouse=True, scope="session")
def segmentation_cells(image_shape):
    seg = np.zeros(image_shape, dtype=np.uint8)
    # assign different values to different corners
    seg[:2, :2, :2] = 1
    seg[-2:, -2:, -2:] = 2
    seg[5:8, 5:8, 5:8] = 3
    return seg


@pytest.fixture(autouse=True, scope="session")
def tmp_coms_csv(shared_tmpdir, voxel_size):
    output_path = shared_tmpdir + "/csvs/"
    os.makedirs(name=output_path, exist_ok=True)
    df = pd.DataFrame(
        {
            "Object ID": [1, 2, 3],
            "COM X (nm)": (np.array([1.1, 3.1, 11.1]) + 0.5) * voxel_size,
            "COM Y (nm)": (np.array([1.1, 3.1, 11.1]) + 0.5) * voxel_size,
            "COM Z (nm)": (np.array([1.1, 3.1, 8.1]) + 0.5) * voxel_size,
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
    # create dataframe
    columns = ["Object ID", "Volume (nm^3)", "Surface Area (nm^2)"]
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
            oi.volume,
            oi.surface_area,
            *oi.com[::-1],
            *oi.bounding_box[:3][::-1],
            *oi.bounding_box[3:][::-1],
        ]
        df.loc[i] = row

    df.to_csv(output_path + "/segmentation_cylinders.csv", index=False)
    return str(output_path + "/segmentation_cylinders.csv")


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
def contact_sites_distance_1(image_shape):
    cs = np.zeros(image_shape, dtype=np.uint8)
    return cs


@pytest.fixture(scope="session")
def contact_sites_distance_2(image_shape):
    cs = np.zeros(image_shape, dtype=np.uint8)
    cs[1:4, 2:5, 4:6] = 1
    cs[1:4, 7:10, 4:8] = 2
    return cs


@pytest.fixture(scope="session")
def contact_sites_distance_3(image_shape):
    cs = np.zeros(image_shape, dtype=np.uint8)
    nonzeros = [
        (0, 2, 4),
        (0, 2, 5),
        (0, 3, 4),
        (0, 3, 5),
        (0, 4, 4),
        (0, 4, 5),
        (0, 7, 4),
        (0, 7, 5),
        (0, 7, 6),
        (0, 7, 7),
        (0, 8, 4),
        (0, 8, 5),
        (0, 8, 6),
        (0, 8, 7),
        (0, 9, 4),
        (0, 9, 5),
        (0, 9, 6),
        (0, 9, 7),
        (1, 2, 2),
        (1, 2, 3),
        (1, 2, 4),
        (1, 2, 5),
        (1, 2, 6),
        (1, 2, 7),
        (1, 3, 2),
        (1, 3, 3),
        (1, 3, 4),
        (1, 3, 5),
        (1, 3, 6),
        (1, 3, 7),
        (1, 4, 2),
        (1, 4, 3),
        (1, 4, 4),
        (1, 4, 5),
        (1, 4, 6),
        (1, 4, 7),
        (1, 5, 2),
        (1, 5, 3),
        (1, 5, 4),
        (1, 5, 5),
        (1, 5, 6),
        (1, 5, 7),
        (1, 5, 8),
        (1, 6, 2),
        (1, 6, 3),
        (1, 6, 4),
        (1, 6, 5),
        (1, 6, 6),
        (1, 6, 7),
        (1, 6, 8),
        (1, 6, 9),
        (1, 7, 2),
        (1, 7, 3),
        (1, 7, 4),
        (1, 7, 5),
        (1, 7, 6),
        (1, 7, 7),
        (1, 7, 8),
        (1, 7, 9),
        (1, 8, 2),
        (1, 8, 3),
        (1, 8, 4),
        (1, 8, 5),
        (1, 8, 6),
        (1, 8, 7),
        (1, 8, 8),
        (1, 8, 9),
        (1, 9, 2),
        (1, 9, 3),
        (1, 9, 4),
        (1, 9, 5),
        (1, 9, 6),
        (1, 9, 7),
        (1, 9, 8),
        (1, 9, 9),
        (2, 2, 3),
        (2, 2, 4),
        (2, 2, 5),
        (2, 2, 6),
        (2, 3, 3),
        (2, 3, 4),
        (2, 3, 5),
        (2, 3, 6),
        (2, 4, 3),
        (2, 4, 4),
        (2, 4, 5),
        (2, 4, 6),
        (2, 5, 4),
        (2, 5, 5),
        (2, 6, 4),
        (2, 6, 5),
        (2, 6, 6),
        (2, 6, 7),
        (2, 7, 3),
        (2, 7, 4),
        (2, 7, 5),
        (2, 7, 6),
        (2, 7, 7),
        (2, 7, 8),
        (2, 8, 3),
        (2, 8, 4),
        (2, 8, 5),
        (2, 8, 6),
        (2, 8, 7),
        (2, 8, 8),
        (2, 9, 3),
        (2, 9, 4),
        (2, 9, 5),
        (2, 9, 6),
        (2, 9, 7),
        (2, 9, 8),
        (3, 2, 4),
        (3, 2, 5),
        (3, 3, 4),
        (3, 3, 5),
        (3, 4, 4),
        (3, 4, 5),
        (3, 7, 4),
        (3, 7, 5),
        (3, 7, 6),
        (3, 7, 7),
        (3, 8, 4),
        (3, 8, 5),
        (3, 8, 6),
        (3, 8, 7),
        (3, 9, 4),
        (3, 9, 5),
        (3, 9, 6),
        (3, 9, 7),
        (4, 2, 4),
        (4, 2, 5),
        (4, 3, 4),
        (4, 3, 5),
        (4, 4, 4),
        (4, 4, 5),
        (4, 7, 4),
        (4, 7, 5),
        (4, 7, 6),
        (4, 7, 7),
        (4, 8, 4),
        (4, 8, 7),
        (4, 9, 4),
        (4, 9, 5),
        (4, 9, 6),
        (4, 9, 7),
    ]
    # for every coordinate in non_zeros, set temp to 1
    for nz in nonzeros:
        cs[nz] = 1
    return cs


import pytest


@pytest.fixture(scope="session")
def shared_tmpdir(tmpdir_factory):
    """Create a shared temporary directory for all test functions."""
    return tmpdir_factory.mktemp("tmp")


@pytest.fixture(scope="session")
def tmp_zarr(shared_tmpdir):
    # do it this way otherwise it appends 0 to the end of the zarr
    output_path = shared_tmpdir + "/tmp.zarr"
    os.makedirs(name=output_path, exist_ok=True)
    return str(output_path)


@pytest.fixture(autouse=True, scope="session")
def tmp_object_information_csv(shared_tmpdir):
    # do it this way otherwise it appends 0 to the end of the zarr
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
    contact_sites_distance_1,
    contact_sites_distance_2,
    contact_sites_distance_3,
):
    dict = {
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
        "contact_sites_distance_1": contact_sites_distance_1,
        "contact_sites_distance_2": contact_sites_distance_2,
        "contact_sites_distance_3": contact_sites_distance_3,
    }
    return dict


@pytest.fixture(autouse=True, scope="session")
def write_zarrs(tmp_zarr, test_image_dict, voxel_size, chunk_size):
    for data_name, data in test_image_dict.items():
        current_voxel_size = (
            voxel_size if "downsampled" not in data_name else 2 * voxel_size
        )
        data_path = f"{tmp_zarr}/{data_name}"

        ds_voxel_size = 3 * [current_voxel_size]
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
            # use larger chunk size for segmentation_cylinders since we had to make it bigger
            ds = create_multiscale_dataset(
                data_path,
                dtype=data.dtype,
                voxel_size=ds_voxel_size,
                total_roi=total_roi,
                write_size=write_size,
            )

        ds.data[:] = data

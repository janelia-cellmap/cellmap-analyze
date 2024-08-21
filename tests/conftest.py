import pytest
import numpy as np
from funlib.geometry import Roi
from cellmap_analyze.util.information_holders import (
    ContactingOrganelleInformation,
    ObjectInformation,
)
from cellmap_analyze.util.zarr_util import create_multiscale_dataset
import os

pytest_plugins = [
    "fixtures.measure",
]


@pytest.fixture(scope="session")
def image_shape():
    return np.array((11, 11, 11))


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
def contact_sites_distance_1(image_shape):
    cs = np.zeros(image_shape, dtype=np.uint64)
    return cs


@pytest.fixture(scope="session")
def contact_sites_distance_2(image_shape):
    cs = np.zeros(image_shape, dtype=np.uint64)
    cs[1:4, 2:5, 4:6] = 1
    cs[1:4, 7:10, 4:8] = 2
    return cs


@pytest.fixture(scope="session")
def contact_sites_distance_3(image_shape):
    cs = np.zeros(image_shape, dtype=np.uint64)
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


@pytest.fixture(scope="session")
def tmp_zarr(tmpdir_factory):
    # do it this way otherwise it appends 0 to the end of the zarr
    base_path = tmpdir_factory.mktemp("tmp")
    output_path = base_path + "/tmp.zarr"
    os.makedirs(name=output_path, exist_ok=True)
    return str(output_path)


@pytest.fixture(scope="session")
def test_image_dict(
    segmentation_1,
    segmentation_2,
    contact_sites_distance_1,
    contact_sites_distance_2,
    contact_sites_distance_3,
):
    dict = {
        "segmentation_1": segmentation_1,
        "segmentation_2": segmentation_2,
        "contact_sites_distance_1": contact_sites_distance_1,
        "contact_sites_distance_2": contact_sites_distance_2,
        "contact_sites_distance_3": contact_sites_distance_3,
    }
    return dict


@pytest.fixture(scope="session")
def voxel_size():
    return 8


@pytest.fixture(autouse=True, scope="session")
def write_zarrs(tmp_zarr, image_shape, test_image_dict, voxel_size):
    for data_name, data in test_image_dict.items():
        data_path = f"{tmp_zarr}/{data_name}"
        ds = create_multiscale_dataset(
            data_path,
            dtype=np.uint8,
            voxel_size=3 * [voxel_size],
            total_roi=Roi((0, 0, 0), image_shape * voxel_size),
            write_size=3 * [4 * voxel_size],
        )
        ds.data[:] = data

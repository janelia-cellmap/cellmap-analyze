import pytest
from cellmap_analyze.analyze.measure import Measure
from cellmap_analyze.util.information_holders import ObjectInformation

import numpy as np


def simple_surface_area(data, voxel_surface_area):
    data = np.pad(data, 1)
    s = data.shape
    sa = 0
    for i in range(1, s[0] - 1):
        for j in range(1, s[1] - 1):
            for k in range(1, s[2] - 1):
                if data[i, j, k]:
                    for delta in [-1, 1]:
                        for dx, dy, dz in [
                            (delta, 0, 0),
                            (0, delta, 0),
                            (0, 0, delta),
                        ]:
                            if data[i + dx, j + dy, k + dz] == 0:
                                sa += 1
    return sa * voxel_surface_area


def simple_contacting_surface_area(contact_site, segmentation, voxel_surface_area):
    s = contact_site.shape
    sa = 0
    for i in range(1, s[0] - 1):
        for j in range(1, s[1] - 1):
            for k in range(1, s[2] - 1):
                if contact_site[i, j, k] and segmentation[i, j, k]:
                    for delta in [-1, 1]:
                        for dx, dy, dz in [
                            (delta, 0, 0),
                            (0, delta, 0),
                            (0, 0, delta),
                        ]:
                            if segmentation[i + dx, j + dy, k + dz] == 0:
                                sa += 1
    return sa * voxel_surface_area


def simple_bounding_box(data, voxel_size):
    x, y, z = np.where(data)

    # add 0.5 to center on voxel
    x = x + 0.5
    y = y + 0.5
    z = z + 0.5

    x *= voxel_size
    y *= voxel_size
    z *= voxel_size
    return [min(x), min(y), min(z), max(x), max(y), max(z)]


def simple_com(data, voxel_size):
    idxs = np.where(data)
    com = (np.mean(idxs, axis=1) + 0.5) * voxel_size
    return com


def simple_id_to_surface_area_dict(contact_site, segmentation, voxel_surface_area):
    id_to_surface_area_dict = {}
    for id in np.unique(segmentation[segmentation > 0]):
        sa = simple_contacting_surface_area(
            contact_site, segmentation == id, voxel_surface_area
        )
        if sa > 0:
            id_to_surface_area_dict[id] = sa
    return id_to_surface_area_dict


def simple_object_information_dict(
    segmentation_1, segmentation_2, contact_site, voxel_size
):
    voxel_volume = voxel_size**3
    voxel_surface_area = voxel_size**2
    object_information_dict = {}
    for id in np.unique(contact_site[contact_site > 0]):
        cs = contact_site == id
        object_information_dict[id] = ObjectInformation(
            volume=np.sum(cs) * voxel_volume,
            com=simple_com(cs, voxel_size),
            surface_area=simple_surface_area(cs, voxel_surface_area),
            bounding_box=simple_bounding_box(cs, voxel_size),
            id_to_surface_area_dict_1=simple_id_to_surface_area_dict(
                cs, segmentation_1, voxel_surface_area
            ),
            id_to_surface_area_dict_2=simple_id_to_surface_area_dict(
                cs, segmentation_2, voxel_surface_area
            ),
        )
    return object_information_dict


@pytest.fixture(scope="session")
def object_information_dict_contact_distance_1(
    segmentation_1, segmentation_2, contact_sites_distance_1, voxel_size
):
    return simple_object_information_dict(
        segmentation_1, segmentation_2, contact_sites_distance_1, voxel_size
    )


@pytest.fixture(scope="session")
def object_information_dict_contact_distance_2(
    segmentation_1, segmentation_2, contact_sites_distance_2, voxel_size
):
    return simple_object_information_dict(
        segmentation_1, segmentation_2, contact_sites_distance_2, voxel_size
    )


@pytest.fixture(scope="session")
def object_information_dict_contact_distance_3(
    segmentation_1, segmentation_2, contact_sites_distance_3, voxel_size
):
    return simple_object_information_dict(
        segmentation_1, segmentation_2, contact_sites_distance_3, voxel_size
    )

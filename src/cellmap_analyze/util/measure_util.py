import numpy as np
from cellmap_analyze.util.information_holders import (
    ContactingOrganelleInformation,
    ObjectInformation,
)

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects
from funlib.evaluate.detection import find_centers
import fastremap


# probably move the following elsewhere
def trim_array(array, trim=1):
    if trim and trim > 0:
        slices = [np.s_[trim:-trim] for _ in range(array.ndim)]
        array = array[tuple(slices)]
    return array


def calculate_surface_areas_voxelwise(
    data: np.ndarray, voxel_face_area=1, do_zero_padding=True
):
    # mirror the edges of the array to calculate the surface areas
    if do_zero_padding:
        data = np.pad(data, 1)

    face_counts = np.zeros_like(data, dtype=int)
    for d in range(data.ndim):
        for delta in [-1, 1]:
            shifted_data = np.roll(data, delta, axis=d)
            face_counts += np.logical_and(data > 0, shifted_data != data)
    surface_areas = face_counts * voxel_face_area

    if do_zero_padding:
        surface_areas = trim_array(surface_areas, 1)

    return surface_areas


def get_surface_areas(data, voxel_face_area=1, mask=None, trim=1):
    surface_areas = calculate_surface_areas_voxelwise(
        data, voxel_face_area, do_zero_padding=(trim == 0)
    )
    # if we have padded the array, need to trim
    if trim:
        surface_areas = trim_array(surface_areas, trim)
        data = trim_array(data, trim)
    if mask:
        mask |= data > 0
    else:
        mask = data > 0

    surface_areas = surface_areas[mask]
    data = data[mask]

    pairs, counts = np.unique(
        np.array([data.ravel(), surface_areas.ravel()]), axis=1, return_counts=True
    )
    voxel_ids = pairs[0]
    voxel_surface_area = pairs[1]
    voxel_counts = counts
    surface_areas_dict = {}
    for voxel_id in np.unique(voxel_ids):
        indices = voxel_ids == voxel_id
        surface_areas_dict[voxel_id] = np.sum(
            voxel_surface_area[indices] * voxel_counts[indices]
        )
    return surface_areas_dict


def get_volumes(data, voxel_volume=1, trim=1):
    if trim:
        data = trim_array(data, trim)
    labels, counts = np.unique(data[data > 0], return_counts=True)
    return dict(zip(labels, counts * voxel_volume))


def get_region_properties(data, voxel_edge_length=1, trim=1):
    voxel_face_area = voxel_edge_length**2
    voxel_volume = voxel_edge_length**3
    surface_areas = get_surface_areas(data, voxel_face_area=voxel_face_area, trim=trim)
    surface_areas = surface_areas.values()
    data = trim_array(data, trim)
    ids, counts = fastremap.unique(data[data > 0], return_counts=True)
    if len(ids) == 0:
        return None
    volumes = counts * voxel_volume
    coms = []
    # coms = np.array(center_of_mass(data, data, index=ids))

    coms = find_centers(data, ids)
    center_on_voxel = 0.5
    coms = np.array(coms) + center_on_voxel

    find_objects_array = data.copy()
    find_objects_ids = list(range(1, len(ids) + 1))
    fastremap.remap(
        find_objects_array,
        dict(zip(ids, find_objects_ids)),
        preserve_missing_labels=True,
        in_place=True,
    )

    bounding_boxes = find_objects(find_objects_array)
    bounding_boxes_coords = []
    for id in find_objects_ids:
        bbox = bounding_boxes[int(id - 1)]
        zmin, ymin, xmin = bbox[0].start, bbox[1].start, bbox[2].start
        zmax, ymax, xmax = bbox[0].stop - 1, bbox[1].stop - 1, bbox[2].stop - 1
        # append to numpy array
        bounding_boxes_coords.append([zmin, ymin, xmin, zmax, ymax, xmax])

    bounding_boxes_coords = np.array(bounding_boxes_coords) + center_on_voxel
    df = pd.DataFrame(
        {
            "ID": ids,
            "Volume (nm^3)": volumes,
            "Surface Area (nm^2)": surface_areas,
            "COM X (nm)": coms[:, 2] * voxel_edge_length,
            "COM Y (nm)": coms[:, 1] * voxel_edge_length,
            "COM Z (nm)": coms[:, 0] * voxel_edge_length,
            "MIN X (nm)": bounding_boxes_coords[:, 2] * voxel_edge_length,
            "MIN Y (nm)": bounding_boxes_coords[:, 1] * voxel_edge_length,
            "MIN Z (nm)": bounding_boxes_coords[:, 0] * voxel_edge_length,
            "MAX X (nm)": bounding_boxes_coords[:, 5] * voxel_edge_length,
            "MAX Y (nm)": bounding_boxes_coords[:, 4] * voxel_edge_length,
            "MAX Z (nm)": bounding_boxes_coords[:, 3] * voxel_edge_length,
        },
    )
    return df


def get_contacting_organelle_information(
    contact_sites, contacting_organelle, voxel_edge_length=1, trim=1
):
    voxel_face_area = voxel_edge_length**2
    surface_areas = calculate_surface_areas_voxelwise(
        contacting_organelle, voxel_face_area
    )

    # trim so we are only considering current block
    surface_areas = trim_array(surface_areas, trim)
    contact_sites = trim_array(contact_sites, trim)
    contacting_organelle = trim_array(contacting_organelle, trim)

    # limit looking to only where contact sites overlap with objects
    mask = np.logical_and(contact_sites > 0, contacting_organelle > 0)
    contact_sites = contact_sites[mask].ravel()
    contacting_organelle = contacting_organelle[mask].ravel()

    surface_areas = surface_areas[mask].ravel()
    groups, counts = np.unique(
        np.array([contact_sites, contacting_organelle, surface_areas]),
        axis=1,
        return_counts=True,
    )
    contact_site_ids = groups[0]
    contacting_ids = groups[1]
    surface_areas = groups[2] * counts
    contact_site_to_contacting_information_dict = {}
    for contact_site_id, contacting_id, surface_area in zip(
        contact_site_ids, contacting_ids, surface_areas
    ):
        coi = contact_site_to_contacting_information_dict.get(
            contact_site_id,
            ContactingOrganelleInformation(),
        )
        coi += ContactingOrganelleInformation({contacting_id: surface_area})
        contact_site_to_contacting_information_dict[contact_site_id] = coi
    return contact_site_to_contacting_information_dict


def get_contacting_organelles_information(
    contact_sites, organelle_1, organelle_2, voxel_edge_length=1, trim=1
):
    contacting_organelle_information_1 = get_contacting_organelle_information(
        contact_sites, organelle_1, voxel_edge_length, trim=trim
    )
    contacting_organelle_information_2 = get_contacting_organelle_information(
        contact_sites, organelle_2, voxel_edge_length, trim=trim
    )
    return contacting_organelle_information_1, contacting_organelle_information_2


def get_object_information(
    object_data, voxel_edge_length, id_offset=0, trim=0, offset=np.zeros((3,)), **kwargs
):
    is_contact_site = False
    if "organelle_1" in kwargs or "organelle_2" in kwargs:
        if "organelle_1" not in kwargs or "organelle_2" not in kwargs:
            raise ValueError(
                "Must provide both organelle_1 and organelle_2 if doing contact site analysis"
            )
        organelle_1 = kwargs.get("organelle_1")
        organelle_2 = kwargs.get("organelle_2")
        is_contact_site = True

    ois = {}
    if np.any(trim_array(object_data, trim)):
        region_props = get_region_properties(
            object_data,
            voxel_edge_length,
            trim=trim,
        )

        if is_contact_site:
            (
                contacting_organelle_information_1,
                contacting_organelle_information_2,
            ) = get_contacting_organelles_information(
                object_data,
                organelle_1,
                organelle_2,
                voxel_edge_length=voxel_edge_length,
                trim=trim,
            )

        # Note some contact site ids may be overwritten but that shouldnt be an issue
        for _, region_prop in region_props.iterrows():

            extra_args = {}
            if is_contact_site:
                extra_args["id_to_surface_area_dict_1"] = (
                    contacting_organelle_information_1.get(
                        region_prop["ID"], ContactingOrganelleInformation()
                    ).id_to_surface_area_dict
                )

                extra_args["id_to_surface_area_dict_2"] = (
                    contacting_organelle_information_2.get(
                        region_prop["ID"], ContactingOrganelleInformation()
                    ).id_to_surface_area_dict
                )

            # need to add global_id_offset here rather than before because region_props find_objects creates an array that is the length of the max id in the array
            id = region_prop["ID"] + id_offset
            ois[id] = ObjectInformation(
                volume=region_prop["Volume (nm^3)"],
                surface_area=region_prop["Surface Area (nm^2)"],
                com=region_prop[["COM Z (nm)", "COM Y (nm)", "COM X (nm)"]].to_numpy()
                + offset,
                bounding_box=[
                    region_prop["MIN Z (nm)"] + offset[0],
                    region_prop["MIN Y (nm)"] + offset[1],
                    region_prop["MIN X (nm)"] + offset[2],
                    region_prop["MAX Z (nm)"] + offset[0],
                    region_prop["MAX Y (nm)"] + offset[1],
                    region_prop["MAX X (nm)"] + offset[2],
                ],
                # if the id is outside of the non-paded crop it wont exist in the following dicts
                **extra_args,
            )
    return ois

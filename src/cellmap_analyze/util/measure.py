import numpy as np
from cellmap_analyze.util.information_holders import (
    ContactingOrganelleInformation,
    ObjectInformation,
)

import numpy as np
from skimage.measure import regionprops_table
import pandas as pd
from scipy.ndimage import find_objects
from funlib.evaluate.detection import find_centers


# probably move the following elsewhere
def trim_array(array, trim=1):
    if trim and trim > 0:
        slices = [np.s_[trim:-trim] for _ in range(array.ndim)]
        array = array[tuple(slices)]
    return array


def calculate_surface_areas_voxelwise(data_padded: np.ndarray, voxel_face_area=1):
    face_counts_padded = np.zeros_like(data_padded)
    for d in range(data_padded.ndim):
        for delta in [-1, 1]:
            shifted_data_padded = np.roll(data_padded, delta, axis=d)
            face_counts_padded += np.logical_and(
                data_padded > 0, shifted_data_padded != data_padded
            )

    surface_areas = face_counts_padded * voxel_face_area
    # surface_areas = trim_array(face_counts_padded, trim) * voxel_face_area
    return surface_areas


def get_surface_areas(data, voxel_face_area=1, mask=None, trim=1):
    surface_areas = calculate_surface_areas_voxelwise(data, voxel_face_area)
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


def get_region_properties(data, voxel_face_area=1, voxel_volume=1, trim=1):
    surface_areas = get_surface_areas(data, voxel_face_area=voxel_face_area, trim=trim)
    surface_areas = surface_areas.values()
    data = trim_array(data, trim)
    ids, counts = np.unique(data[data > 0], return_counts=True)
    if len(ids) == 0:
        return None
    volumes = counts * voxel_volume
    coms = []
    # coms = np.array(center_of_mass(data, data, index=ids))

    coms = find_centers(data, ids)
    coms = np.array(coms)
    bounding_boxes = find_objects(data)
    bounding_boxes_coords = []
    for id in ids:
        bbox = bounding_boxes[int(id - 1)]
        zmin, ymin, xmin = bbox[0].start, bbox[1].start, bbox[2].start
        zmax, ymax, xmax = bbox[0].stop, bbox[1].stop, bbox[2].stop
        # append to numpy array
        bounding_boxes_coords.append([zmin, ymin, xmin, zmax, ymax, xmax])

    bounding_boxes_coords = np.array(bounding_boxes_coords)
    df = pd.DataFrame(
        {
            "ID": ids,
            "Volume (nm^3)": volumes,
            "Surface Area (nm^2)": surface_areas,
            "COM X (nm)": coms[:, 2],
            "COM Y (nm)": coms[:, 1],
            "COM Z (nm)": coms[:, 0],
            "MIN X (nm)": bounding_boxes_coords[:, 2],
            "MIN Y (nm)": bounding_boxes_coords[:, 1],
            "MIN Z (nm)": bounding_boxes_coords[:, 0],
            "MAX X (nm)": bounding_boxes_coords[:, 5],
            "MAX Y (nm)": bounding_boxes_coords[:, 4],
            "MAX Z (nm)": bounding_boxes_coords[:, 3],
        },
    )
    return df


def get_contacting_organelle_information(
    contact_sites, contacting_organelle, voxel_face_area=1, trim=0
):

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
    contact_sites, organelle_1, organelle_2, trim=0
):
    contacting_organelle_information_1 = get_contacting_organelle_information(
        contact_sites, organelle_1, trim=trim
    )
    contacting_organelle_information_2 = get_contacting_organelle_information(
        contact_sites, organelle_2, trim=trim
    )
    return contacting_organelle_information_1, contacting_organelle_information_2


def get_object_information(
    object_data, voxel_face_area, voxel_volume, padding_voxels, id_offset=0, **kwargs
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
    if np.any(object_data):
        region_props = get_region_properties(
            object_data,
            voxel_face_area,
            voxel_volume,
            trim=padding_voxels,
        )

        if is_contact_site:
            (
                contacting_organelle_information_1,
                contacting_organelle_information_2,
            ) = get_contacting_organelles_information(
                object_data,
                organelle_1,
                organelle_2,
                trim=padding_voxels,
            )
            # # # current_contact_site_ids = np.unique(contact_sites[contact_sites > 0])

        # Note some contact site ids may be overwritten but that shouldnt be an issue
        for _, region_prop in region_props.iterrows():

            extra_args = {}
            if is_contact_site:
                extra_args["id_to_surface_area_dict_1"] = (
                    contacting_organelle_information_1.get(region_prop["ID"], {})
                )
                extra_args["id_to_surface_area_dict_2"] = (
                    contacting_organelle_information_2.get(region_prop["ID"], {})
                )

            # need to add global_id_offset here rather than before because region_props find_objects creates an array that is the length of the max id in the array
            id = region_prop["ID"] + id_offset
            ois[id] = ObjectInformation(
                volume=region_prop["Volume (nm^3)"],
                surface_area=region_prop["Surface Area (nm^2)"],
                com=region_prop[["COM Z (nm)", "COM Y (nm)", "COM X (nm)"]].to_numpy(),
                # if the id is outside of the non-paded crop it wont exist in the following dicts
                **extra_args,
            )
    return ois

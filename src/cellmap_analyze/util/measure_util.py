# %%
import numpy as np
from cellmap_analyze.util.information_holders import (
    ContactingOrganelleInformation,
    ObjectInformation,
)

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects
from cellmap_analyze.cythonizing.centers import find_centers
import fastremap


# probably move the following elsewhere
def trim_array(array, trim=1):
    if trim and trim > 0:
        slices = [np.s_[trim:-trim] for _ in range(array.ndim)]
        array = array[tuple(slices)]
    return array


# chatgpt
def calculate_surface_areas_voxelwise(
    data: np.ndarray, voxel_face_area: float = 1, do_zero_padding: bool = True
) -> np.ndarray:
    # 1) Optionally pad with zeros (so we never have wrap or missing‐neighbor)
    if do_zero_padding:
        pad_width = [(1, 1)] * data.ndim
        data = np.pad(data, pad_width, mode="constant", constant_values=0)

    # 2) Prepare face_counts buffer
    face_counts = np.zeros_like(data, dtype=int)
    ndim = data.ndim

    # 3) For each axis, compare forward + backward with distinct slices
    for axis in range(ndim):
        # forward neighbor: orig [:-1], neigh [1:]
        orig_fwd = [slice(None)] * ndim
        neigh_fwd = [slice(None)] * ndim
        orig_fwd[axis] = slice(None, -1)
        neigh_fwd[axis] = slice(1, None)
        of, nf = tuple(orig_fwd), tuple(neigh_fwd)

        mask_fwd = (data[of] > 0) & (data[of] != data[nf])
        face_counts[of] += mask_fwd

        # backward neighbor: orig [1:], neigh [:-1]
        orig_bwd = [slice(None)] * ndim
        neigh_bwd = [slice(None)] * ndim
        orig_bwd[axis] = slice(1, None)
        neigh_bwd[axis] = slice(None, -1)
        ob, nb = tuple(orig_bwd), tuple(neigh_bwd)

        mask_bwd = (data[ob] > 0) & (data[ob] != data[nb])
        face_counts[ob] += mask_bwd

    # 4) Multiply by face area
    surface_areas = face_counts * voxel_face_area

    # 5) If we padded, trim off the 1-voxel border
    if do_zero_padding:
        trim_slices = tuple(slice(1, -1) for _ in range(ndim))
        surface_areas = surface_areas[trim_slices]

    return surface_areas


# chatgpt
def get_surface_areas(
    data: np.ndarray, voxel_face_area: float = 1, mask: np.ndarray = None, trim: int = 1
):
    # 1) per‐voxel SAs
    surface_areas = calculate_surface_areas_voxelwise(
        data, voxel_face_area, do_zero_padding=(trim == 0)
    )

    # 2) optional trim
    if trim:
        surface_areas = trim_array(surface_areas, trim)
        data = trim_array(data, trim)

    # 3) build final mask
    if mask is None:
        final_mask = data > 0
    else:
        final_mask = mask | (data > 0)

    # 4) gather labels & SAs
    labels = data[final_mask].ravel()
    sa_vals = surface_areas[final_mask].ravel()

    if labels.size == 0:
        return {}

    # 5) group and sum without huge arrays:
    #    - unique_labels: sorted unique IDs (still uint64)
    #    - inverse: map each label → index in unique_labels
    unique_labels, inverse = fastremap.unique(labels, return_inverse=True)
    #    - sums[i] = total SA for unique_labels[i]
    sums = np.bincount(inverse, weights=sa_vals)

    # 6) build output dict
    #    cast keys back to Python int if you like, or keep as uint64
    return {int(lbl): float(sums[idx]) for idx, lbl in enumerate(unique_labels)}


def get_volumes(data, voxel_volume=1, trim=1):
    if trim:
        data = trim_array(data, trim)
    labels, counts = np.unique(data[data > 0], return_counts=True)
    return dict(zip(labels, counts * voxel_volume))


def get_region_properties(data, voxel_edge_length=1, trim=1):
    voxel_face_area = voxel_edge_length**2
    voxel_volume = voxel_edge_length**3
    surface_areas = get_surface_areas(data, voxel_face_area=voxel_face_area, trim=trim)
    data = trim_array(data, trim)
    ids, counts = fastremap.unique(data[data > 0], return_counts=True)
    # need this to get in correct order due to "bug" in fastremap: https://github.com/seung-lab/fastremap/issues/42
    surface_areas = np.array([surface_areas[id] for id in ids])
    if len(ids) == 0:
        return None
    volumes = counts * voxel_volume
    coms = []
    # coms = np.array(center_of_mass(data, data, index=ids))

    coms, sum_r2 = find_centers(data, ids, compute_sum_r2=True)
    #TODO: do stuff with sum_r2
    add 0.5 to sum_r2...
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
# %%
import numpy as np
seg = np.random.randint(0, 5, size=(212, 212, 212))  # Example segmentation array
%timeit find_centers(seg,[1,2,3,4], compute_sum_r2=True)
# # seg: your (nx,ny,nz) integer array of labels (0=background)
# def calc(seg):
#     # 1) Precompute r² weights
#     X, Y, Z = np.indices(seg.shape, dtype=np.int64)
#     r2 = (X*X + Y*Y + Z*Z).ravel()

#     # 2) Flatten labels and mask out zeros
#     labels = seg.ravel()
#     mask   = labels != 0

#     # 3) Bin only the nonzero labels
#     sum_r2 = np.bincount(labels[mask], weights=r2[mask])
#     # sum_r2[i-1] is now ∑‖r‖² for object label i, for i=1…max(seg)

#     # If you want it aligned so that sum_r2_nonzero[i] is for label i:
#     maxlabel = seg.max()
#     sum_r2_full = np.zeros(maxlabel+1, dtype=r2.dtype)
#     sum_r2_full[1:] = sum_r2      # index 0 stays zero
#     # sum_r2_full[L] gives ∑‖r‖² for label L; sum_r2_full[0] == 0
#     return sum_r2
# %%
%timeit calc(seg)
# %%
1+2
# %%

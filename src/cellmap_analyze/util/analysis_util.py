import logging
import numpy as np
from skimage.measure import regionprops_table
import pandas as pd
from scipy.ndimage import find_objects, center_of_mass
from cellmap_analyze.util.io_util import print_with_datetime

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    print_with_datetime("surface_areas done", logger)
    ids, counts = np.unique(data[data > 0], return_counts=True)
    if len(ids) == 0:
        return None
    volumes = counts * voxel_volume
    coms = []
    # coms = np.array(center_of_mass(data, data, index=ids))
    print_with_datetime("coms", logger)
    for id in ids:
        coms.append(np.mean(np.argwhere(data == id), axis=0))
        # coms.append(center_of_mass(data == id))
    coms = np.array(coms)
    print_with_datetime(data.dtype, logger)
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
    # create dataframe with values

    # print_with_datetime("trimmed", logger)
    # region_props = regionprops_table(
    #     data, properties=["label", "area", "centroid", "bbox"], cache=False
    # )
    # print_with_datetime("regionprops done", logger)
    # df = pd.DataFrame(region_props)
    # df.rename(columns={"label": "id"}, inplace=True)

    # # add surface area column to df
    # df["surface_area"] = df["id"].map(surface_areas)

    # # concatenate columns named com-0, com-1, com-2 to numpy array
    # df["com"] = df["id"].map(
    #     dict(
    #         zip(
    #             df["id"].tolist(),
    #             np.array(df[["centroid-0", "centroid-1", "centroid-2"]].values),
    #         )
    #     )
    # )
    # # do the same but now to list
    # df["bounding_box"] = df[
    #     ["bbox-0", "bbox-1", "bbox-2", "bbox-3", "bbox-4", "bbox-5"]
    # ].values.tolist()
    # # remove centroid and bbox columns
    # df = df.drop(
    #     columns=[
    #         "centroid-0",
    #         "centroid-1",
    #         "centroid-2",
    #         "bbox-0",
    #         "bbox-1",
    #         "bbox-2",
    #         "bbox-3",
    #         "bbox-4",
    #         "bbox-5",
    #     ]
    # )
    # rename area to volume
    # df.rename(columns={"area": "volume"}, inplace=True)
    # df["volume"] *= voxel_volume
    # return df

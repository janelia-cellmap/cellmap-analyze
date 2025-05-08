# %%
import struct
import os
import struct
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import scipy.spatial as spatial


def write_precomputed_annotations(
    annotation_type, annotations, output_directory, densities
):
    if os.path.exists(f"{output_directory}/spatial0"):
        os.system(f"rm -rf {output_directory}/spatial0")
    # if os.path.exists(f"{output_directory}/relationships"):
    #     os.system(f"rm -rf {output_directory}/relationships")
    os.makedirs(f"{output_directory}/spatial0", exist_ok=True)
    # os.makedirs(f"{output_directory}/relationships", exist_ok=True)

    if annotation_type == "line":
        coords_to_write = 6
    else:
        coords_to_write = 3

    annotations_and_densities = list(zip(annotations, densities))
    with open(f"{output_directory}/spatial0/0_0_0", "wb") as outfile:
        total_count = len(annotations)
        buf = struct.pack("<Q", total_count)
        for annotation, density in tqdm(annotations_and_densities):
            annotation_buf = struct.pack(f"<{coords_to_write}f", *annotation)
            buf += annotation_buf
            buf += struct.pack(f"<1f", np.float32(density))  # property

        # write the ids at the end of the buffer as increasing integers
        id_buf = struct.pack(
            f"<{total_count}Q", *range(1, len(annotations) + 1, 1)
        )  # so start at 1
        # id_buf = struct.pack('<%sQ' % len(coordinates), 3,1 )#s*range(len(coordinates)))
        buf += id_buf
        outfile.write(buf)

    # with open(f"{output_directory}/relationships/1", "wb") as outfile:
    #     # annotations = annotations[:10000]
    #     # densities = densities[:10000]
    #     annotations_and_densities = list(zip(annotations, densities))
    #     total_count = len(annotations_and_densities)
    #     buf = struct.pack("<Q", total_count)
    #     for annotation, density in tqdm(annotations_and_densities):
    #         annotation_buf = struct.pack(f"<{coords_to_write}f", *annotation)
    #         buf += annotation_buf
    #         buf += struct.pack(f"<1f", np.float32(density))  # property

    #     # write the ids at the end of the buffer as increasing integers
    #     id_buf = struct.pack(
    #         f"<{total_count}Q", *range(1, len(annotations) + 1, 1)
    #     )  # so start at 1
    #     # id_buf = struct.pack('<%sQ' % len(coordinates), 3,1 )#s*range(len(coordinates)))
    #     buf += id_buf
    #     outfile.write(buf)

    max_extents = annotations.reshape((-1, 3)).max(axis=0) + 1
    max_extents = [int(max_extent) for max_extent in max_extents]
    info = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {"z": [1, "nm"], "y": [1, "nm"], "x": [1, "nm"]},
        "by_id": {"key": "by_id"},
        "lower_bound": [0, 0, 0],
        "upper_bound": max_extents,
        "annotation_type": annotation_type,
        "properties": [{"id": "density", "type": "float32", "description": "density"}],
        "relationships": [],  # {"id": "associated_column_cell", "key": "relationships"}],
        "spatial": [
            {
                "chunk_size": max_extents,
                "grid_shape": [1, 1, 1],
                "key": "spatial0",
                "limit": 1,
            }
        ],
    }

    with open(f"{output_directory}/info", "w") as info_file:
        json.dump(info, info_file)

    return output_directory


def calculate_densities(
    measurement_file, output_directory, radius_nm=1000, minimum_volume_nm_3=None
):
    organelles = pd.read_csv(measurement_file)
    com_x = organelles["COM X (nm)"].to_numpy()
    com_y = organelles["COM Y (nm)"].to_numpy()
    com_z = organelles["COM Z (nm)"].to_numpy()

    coms = np.column_stack((com_z, com_y, com_x))
    if minimum_volume_nm_3:
        volumes = organelles["Volume (nm^3)"].to_numpy()
        idxs_to_keep = np.where(volumes >= minimum_volume_nm_3)[0]
        coms = coms[idxs_to_keep]

    tree = spatial.KDTree(coms)
    neighbors = tree.query_ball_tree(tree, radius_nm)
    densities = np.array([len(n) for n in neighbors])
    # densities = densities / densities.max()

    # # write_precomputed_annotations("line", annotations)
    # # write_precomputed_annotations(
    # #     "line",
    # #     annotations,
    # #     densities,
    # # )

    return write_precomputed_annotations(
        "point",
        coms,
        output_directory,
        densities,
    )


# %%
for dataset in [
    "jrc_22ak351-leaf-3m",
    # "jrc_22ak351-leaf-3r",
    # "jrc_22ak351-leaf-2l",
    # "jrc_22ak351-leaf-3mb",
    # "jrc_22ak351-leaf-3rb",
    # "jrc_22ak351-leaf-2lb",
]:
    output_annotations_path = calculate_densities(
        f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{dataset}/plasmodesmata_cleaned.csv",
        f"/nrs/cellmap/ackermand/cellmap/plasmodesmata/neuroglancer_annotations/{dataset}/densities",
    )
    print(output_annotations_path)
# %%
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

for dataset in [
    "jrc_22ak351-leaf-3mb",
    "jrc_22ak351-leaf-3rb",
    "jrc_22ak351-leaf-2lb",
]:
    print(dataset)
    df = pd.read_csv(
        f"/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{dataset}/plasmodesmata.csv"
    )
    coms = df[["COM X (nm)", "COM Y (nm)", "COM Z (nm)"]].values
    tree = cKDTree(coms)
    pairs = tree.query_pairs(r=5_000)
    # Compute distances for these pairs:
    sample_distances = np.array([np.linalg.norm(coms[i] - coms[j]) for i, j in pairs])
    import matplotlib.pyplot as plt

    bins = np.arange(0, 5_000 + 500, 500)
    counts, bin_edges = np.histogram(sample_distances, bins=bins)

    # Calculate the volume of a sphere for each bin using the maximum radius of the bin
    # r_max is the upper edge of each bin
    r_max = bin_edges[1:]
    volumes = (4 / 3) * np.pi * (r_max**3)
    normalized_counts = counts / volumes

    # Plot the normalized histogram
    plt.bar(bin_edges[:-1], normalized_counts, width=500, align="edge")
    # plt.ylim(0, 0.0015)
    plt.xlabel("Distance (nm)")
    plt.ylabel("Normalized count (per volume)")
    plt.title("Histogram of Pairwise Distances Normalized by Volume")
    plt.show()

# %%
import numpy as np
import scipy.ndimage as ndi

import numpy as np
import scipy.ndimage as ndi


def combine_multiresolution_native(images, voxel_sizes, final_shape=None):
    """
    Combine several 3D images (all of the same matrix shape) that have different
    physical extents (due to differing voxel sizes) into one composite volume.

    The images are assumed to be centered. For each image, we define its valid region
    as a sphere with radius:

         R_i = (min(image.shape) / 2) * voxel_sizes[i]

    The composite volume is defined on a grid of shape final_shape (defaulting to the
    shape of the lowest-resolution image) and uses the lowest-resolution voxel size to
    convert voxel indices to physical coordinates.

    For each composite voxel (with physical coordinate computed relative to the center),
    we determine which image has valid data there:
      - If r < R_0 (high-res effective radius), sample the high-res image.
      - Else if r is between R_0 and R_1 (mid-res effective radius), sample the mid-res.
      - Else if r is between R_1 and R_2, use the next image, etc.

    Voxels outside the outermost image’s effective region will remain zero.

    Parameters:
      images : list of np.ndarray
          3D arrays ordered from highest resolution (level 0) to lowest resolution.
      voxel_sizes : list of floats
          Voxel sizes (in physical units) for each image (assumed isotropic).
      final_shape : tuple of ints, optional
          Desired shape of the composite volume. If None, uses the shape of the lowest
          resolution image (i.e. images[-1].shape).

    Returns:
      composite : np.ndarray
          The composite volume that uses high-res data in the center, then progressively
          lower resolution data outward.
    """
    num_levels = len(images)
    if num_levels == 0:
        raise ValueError("No images provided.")

    # Use the lowest-resolution image's shape if not provided.
    if final_shape is None:
        final_shape = images[-1].shape

    # Use the lowest-resolution voxel size to define the composite grid.
    comp_voxel_size = voxel_sizes[-1]
    comp_center = (np.array(final_shape) - 1) / 2.0

    # Create a coordinate grid (in voxel indices) for the composite volume.
    zz, yy, xx = np.indices(final_shape)
    # Convert voxel indices to physical coordinates (using the composite voxel size)
    # relative to the composite center.
    x_phys = (xx - comp_center[2]) * comp_voxel_size
    y_phys = (yy - comp_center[1]) * comp_voxel_size
    z_phys = (zz - comp_center[0]) * comp_voxel_size
    radii = np.sqrt(x_phys**2 + y_phys**2 + z_phys**2)

    # Compute effective physical radii for each image.
    effective_radii = []
    for i, img in enumerate(images):
        shape = np.array(img.shape)
        effective_radii.append((min(shape) / 2.0) * voxel_sizes[i])

    # Initialize composite volume.
    composite = np.zeros(final_shape, dtype=images[0].dtype)

    # For each level, sample the image in its valid region.
    for level in range(num_levels):
        if level == 0:
            mask = radii < effective_radii[0]
        else:
            mask = (radii >= effective_radii[level - 1]) & (
                radii < effective_radii[level]
            )
        if not np.any(mask):
            continue

        img = images[level]
        v_img = voxel_sizes[level]
        # Assume the image is centered at (shape - 1)/2 in its native voxel coordinates.
        img_center = (np.array(img.shape) - 1) / 2.0

        # Get the composite physical coordinates for the selected voxels.
        x_sel = x_phys[mask]
        y_sel = y_phys[mask]
        z_sel = z_phys[mask]

        # Map these physical coordinates to the native index space of the current image.
        # Formula: image_index = (physical_coordinate / v_img) + img_center.
        x_img = x_sel / v_img + img_center[2]
        y_img = y_sel / v_img + img_center[1]
        z_img = z_sel / v_img + img_center[0]

        coords = np.vstack([z_img, y_img, x_img])
        sampled = ndi.map_coordinates(img, coords, order=1, mode="constant", cval=0.0)
        composite[mask] = sampled

    return composite


from cellmap_analyze.util.image_data_interface import ImageDataInterface
from funlib.geometry import Roi

images = []
center = (np.array([97800, 49000, 112000]) // 8) * 8
shape = 512
for scale in range(5):
    resolution = 8 * 2**scale
    roi_start = center - (shape // 2 * resolution)
    roi_shape = (shape * resolution, shape * resolution, shape * resolution)
    data = ImageDataInterface(
        f"/nrs/cellmap/data/jrc_mus-liver-zon-2/jrc_mus-liver-zon-2.zarr/recon-1/em/fibsem-uint8/s{scale}",
    ).to_ndarray_ts(Roi(roi_start, roi_shape))
    images.append(data)

# %%
composite = combine_multiresolution_native(images, voxel_sizes=[8, 16, 32, 64, 128])

print("Composite image shape:", composite.shape)
# %%
import matplotlib.pyplot as plt

plt.imshow(composite[256, :, :])

# %%
roi_shape[0] / 8**5
# %%

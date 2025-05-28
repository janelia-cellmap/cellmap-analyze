import numpy as np
import neuroglancer
import os
import struct
import json
from .image_data_interface import ImageDataInterface


def view_in_neuroglancer(**kwargs):  # pragma: no cover
    # get variable name as string
    neuroglancer.set_server_bind_address("0.0.0.0")
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        for array_name, array in kwargs.items():

            if type(array) != np.ndarray:
                array = ImageDataInterface(array).to_ndarray_ts()

            if (
                array.dtype in (float, np.float32)
                or "raw" in array_name
                or "__img" in array_name
            ):
                s.layers[array_name] = neuroglancer.ImageLayer(
                    source=neuroglancer.LocalVolume(
                        data=array,
                    ),
                )
            else:
                s.layers[array_name] = neuroglancer.SegmentationLayer(
                    source=neuroglancer.LocalVolume(
                        data=array,
                    ),
                )

    print(viewer.get_viewer_url())


def write_out_annotations(output_directory, annotation_ids, annotations):
    annotation_type = "line"
    os.makedirs(f"{output_directory}/spatial0", exist_ok=True)

    total_count = len(annotations)
    # Create header with total count
    header = struct.pack("<Q", total_count)

    # Convert annotations list to a NumPy array and then to bytes
    annotations_np = np.array(annotations, dtype=np.float32)
    annotations_bytes = annotations_np.tobytes()

    # Create an array for IDs starting at 1 and convert it to bytes
    id_buf = annotation_ids.tobytes()

    # Write the combined buffer to file
    with open(f"{output_directory}/spatial0/0_0_0", "wb") as outfile:
        outfile.write(header + annotations_bytes + id_buf)

    max_extents = annotations.reshape((-1, 3)).max(axis=0) + 1
    max_extents = [int(max_extent) for max_extent in max_extents]
    info = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {"z": [1, "nm"], "y": [1, "nm"], "x": [1, "nm"]},
        "by_id": {"key": "by_id"},
        "lower_bound": [0, 0, 0],
        "upper_bound": max_extents,
        "annotation_type": annotation_type,
        "properties": [],
        "relationships": [],
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

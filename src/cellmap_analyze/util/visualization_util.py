import numpy as np
import neuroglancer


def view_in_neuroglancer(**kwargs):
    # get variable name as string
    neuroglancer.set_server_bind_address("0.0.0.0")
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        for array_name, array in kwargs.items():
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

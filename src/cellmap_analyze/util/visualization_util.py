import neuroglancer


def view_in_neuroglancer(**kwargs):
    # get variable name as string

    neuroglancer.set_server_bind_address("0.0.0.0")
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        for array_name, array in kwargs.items():
            s.layers[array_name] = neuroglancer.SegmentationLayer(
                source=neuroglancer.LocalVolume(
                    data=array,
                ),
            )
    return viewer.get_viewer_url()

import neuroglancer
import numpy as np

if __name__ == "__main__":
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.layers["test"] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(
                data=np.random.rand(100, 100, 100)
                )
            )

    print("Viewer URL:", viewer.get_viewer_url())
    input("Press Enter to close the viewers...")

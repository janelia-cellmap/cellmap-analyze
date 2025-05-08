import numpy as np
from funlib.geometry import Roi
from cellmap_analyze.util.image_data_interface import ImageDataInterface
idi = ImageDataInterface(
    "/nrs/cellmap/ackermand/predictions/cellmap_experiments/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.zarr/predictions/2025-02-15_3m/plasmodesmata_affs_lsds/0__affs"
)
1402, 3818, 1336, 176, 101, 510
56816, 41197, 2076, 176, 101, 510
roi = Roi(np.array([2076, 41197, 56816]), 3*[8*216])


from scipy.ndimage import measurements, gaussian_filter
import mwatershed as mws
import fastremap


def filter_fragments(
    affs_data: np.ndarray, fragments_data: np.ndarray, filter_val: float
) -> None:
    """Allows filtering of MWS fragments based on mean value of affinities & fragments. Will filter and update the fragment array in-place.

    Args:
        aff_data (``np.ndarray``):
            An array containing affinity data.

        fragments_data (``np.ndarray``):
            An array containing fragment data.

        filter_val (``float``):
            Threshold to filter if the average value falls below.
    """

    average_affs: float = np.mean(affs_data.data, axis=0)

    filtered_fragments: list = []

    fragment_ids: np.ndarray = np.unique(fragments_data)

    for fragment, mean in zip(
        fragment_ids, measurements.mean(average_affs, fragments_data, fragment_ids)
    ):
        if mean < filter_val:
            filtered_fragments.append(fragment)

    filtered_fragments: np.ndarray = np.array(
        filtered_fragments, dtype=fragments_data.dtype
    )
    # replace: np.ndarray = np.zeros_like(filtered_fragments)
    fastremap.mask(fragments_data, filtered_fragments, in_place=True)


neighborhood = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [3, 0, 0],
    [0, 3, 0],
    [0, 0, 3],
    [9, 0, 0],
    [0, 9, 0],
    [0, 0, 9],
]
# neighborhood = [[0,0,1],[0,1,0],[1,0,0],[0,0,3],[0,3,0],[3,0,0],[0,0,9],[0,9,0],[9,0,0]]
import time


def mutex_watershed_blockwise(
    data, adjacent_edge_bias=-0.4, lr_bias_ratio=-0.08, filter_val=0.5
):
    if data.dtype == np.uint8:
        logger.info("Assuming affinities are in [0,255]")
        max_affinity_value: float = 255.0
        data = data.astype(np.float64)
    else:
        data = data.astype(np.float64)
        max_affinity_value: float = 1.0

    data /= max_affinity_value

    if data.max() < 1e-3:
        segmentation = np.zeros(data.shape, dtype=np.uint64)
        return data

    t0 = time.time()
    random_noise = np.random.randn(*data.shape) * 0.001
    smoothed_affs = (
        gaussian_filter(data, sigma=(0, *(np.amax(neighborhood, axis=0) / 3))) - 0.5
    ) * 0.01
    shift: np.ndarray = np.array(
        [
            (
                adjacent_edge_bias
                if max(offset) <= 1
                else np.linalg.norm(offset) * lr_bias_ratio
            )
            for offset in neighborhood
        ]
    ).reshape((-1, *((1,) * (len(data.shape) - 1))))

    # raise Exception(data.max(), data.min(), self.neighborhood)

    # segmentation = mws.agglom(
    #     data.astype(np.float64) - self.bias,
    #     self.neighborhood,
    # )

    # filter fragments
    t1 = time.time()
    segmentation = mws.agglom(
        data + shift + random_noise + smoothed_affs,
        offsets=neighborhood,
    )
    t2 = time.time()

    if filter_val > 0.0:
        filter_fragments(data, segmentation, filter_val)
    t3 = time.time()
    # fragment_ids = fastremap.unique(segmentation[segmentation > 0])
    # fastremap.mask_except(segmentation, filtered_fragments, in_place=True)
    fastremap.renumber(segmentation, in_place=True)
    t4 = time.time()
    print(f"{t1-t0},{t2-t1}, {t3-t2}, {t4-t3}")
    return segmentation


import cc3d


def instancewise_instance_segmentation(segmentation):
    ids = fastremap.unique(segmentation[segmentation > 0])
    output = np.zeros_like(segmentation, dtype=np.uint64)
    for id in ids:
        cc = cc3d.connected_components(
            segmentation == id,
            connectivity=6,
            binary_image=True,
        )
        cc[cc > 0] += np.max(output)
        output += cc
    return output


res = idi.to_ndarray_ts(roi)
output = mutex_watershed_blockwise(res)

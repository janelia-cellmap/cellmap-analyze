# %%
import logging
from pathlib import Path
import tensorstore as ts
import numpy as np
from funlib.geometry import Coordinate
from funlib.geometry import Roi

from cellmap_analyze.util.io_util import split_dataset_path
from funlib.persistence import open_ds
from skimage.measure import block_reduce

# Much below taken from flyemflows: https://github.com/janelia-flyem/flyemflows/blob/master/flyemflows/util/util.py
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def open_ds_tensorstore(dataset_path: str, mode="r", concurrency_limit=None):
    # open with zarr or n5 depending on extension
    filetype = (
        "zarr" if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5") else "n5"
    )
    if concurrency_limit:
        spec = {
            "driver": filetype,
            "context": {
                "data_copy_concurrency": {"limit": concurrency_limit},
                "file_io_concurrency": {"limit": concurrency_limit},
            },
            "kvstore": {
                "driver": "file",
                "path": dataset_path,
            },
        }
    else:
        spec = {
            "driver": filetype,
            "kvstore": {
                "driver": "file",
                "path": dataset_path,
            },
        }

    if mode == "r":
        dataset_future = ts.open(spec, read=True, write=False)
    else:
        dataset_future = ts.open(spec, read=False, write=True)

    return dataset_future.result()


def to_ndarray_tensorstore(
    dataset,
    roi=None,
    voxel_size=None,
    offset=None,
    output_voxel_size=None,
    swap_axes=False,
    custom_fill_value=None,
):
    """Read a region of a tensorstore dataset and return it as a numpy array

    Args:
        dataset ('tensorstore.dataset'): Tensorstore dataset
        roi ('funlib.geometry.Roi'): Region of interest to read

    Returns:
        Numpy array of the region
    """

    if output_voxel_size is None:
        output_voxel_size = voxel_size

    rescale_factor = voxel_size[0] / output_voxel_size[0]

    if swap_axes:
        print("Swapping axes")
        if roi:
            roi = Roi(roi.begin[::-1], roi.shape[::-1])
        if offset:
            offset = Coordinate(offset[::-1])

    channel_offset = 0
    domain = dataset.domain
    if len(domain) > 3:
        # Determine how many dimensions to skip (channels dimension exists if channels > 0)
        channel_offset = 1
        channels = slice(domain[0].inclusive_min, domain[0].exclusive_max)
        domain = domain[1:]

    if roi is None:
        with ts.Transaction() as txn:
            data = dataset.with_transaction(txn).read().result()
        if rescale_factor > 1:
            data = (
                data.repeat(rescale_factor, axis=0 + channel_offset)
                .repeat(rescale_factor, 1 + channel_offset)
                .repeat(rescale_factor, 2 + channel_offset)
            )
        return data

    if offset is None:
        offset = Coordinate(np.zeros(roi.dims, dtype=int))

    if voxel_size != output_voxel_size:
        # in the case where there is a mismatch in voxel sizes, we may need to extra pad to ensure that the output is a multiple of the output voxel size
        original_roi = roi
        roi = original_roi.snap_to_grid(voxel_size)
        snapped_offset = (original_roi.begin - roi.begin) / output_voxel_size
        snapped_end = (original_roi.end - roi.begin) / output_voxel_size
        snapped_slices = tuple(
            slice(snapped_offset[i], snapped_end[i]) for i in range(3)
        )

    roi = roi.snap_to_grid(voxel_size)
    roi -= offset
    roi /= voxel_size

    # in the event that we are passing things at half voxel offsets, we need to snap the roi to the grid

    # Specify the range
    roi_slices = roi.to_slices()

    # Compute the valid range
    valid_slices = tuple(
        slice(max(s.start, inclusive_min), min(s.stop, exclusive_max))
        for s, inclusive_min, exclusive_max in zip(
            roi_slices, domain.inclusive_min, domain.exclusive_max
        )
    )

    pad_width = [
        [valid_slice.start - s.start, s.stop - valid_slice.stop]
        for s, valid_slice in zip(roi_slices, valid_slices)
    ]

    if channel_offset > 0:
        pad_width = [[0, 0]] + pad_width
        valid_slices = (channels,) + valid_slices

    # Create an array to hold the requested data, filled with a default value (e.g., zeros)
    # output_shape = [s.stop - s.start for s in roi_slices]
    # valid_slices = (slice(None),) + valid_slices, channel stuff
    if not dataset.fill_value:
        fill_value = 0
    if custom_fill_value:
        fill_value = custom_fill_value
    with ts.Transaction() as txn:
        data = dataset.with_transaction(txn)[valid_slices].read().result()

    if np.any(np.array(pad_width)):
        if fill_value == "edge":
            data = np.pad(
                data,
                pad_width=pad_width,
                mode="edge",
            )
        else:
            data = np.pad(
                data,
                pad_width=pad_width,
                mode="constant",
                constant_values=fill_value,
            )
    # else:
    #     padded_data = (
    #         np.ones(output_shape, dtype=dataset.dtype.numpy_dtype) * fill_value
    #     )
    #     padded_slices = tuple(
    #         slice(valid_slice.start - s.start, valid_slice.stop - s.start)
    #         for s, valid_slice in zip(roi_slices, valid_slices)
    #     )

    #     # Read the region of interest from the dataset
    #     padded_data[padded_slices] = dataset[valid_slices].read().result()

    # Create a slicing tuple that preserves the channel dimension (if present) and applies snapped_slices to the spatial axes
    if rescale_factor != 1:
        slices = (slice(None),) * channel_offset + snapped_slices
        if rescale_factor > 1:
            # Compute the upsampling factor based on the first spatial dimension
            factor = voxel_size[0] / output_voxel_size[0]
            # Upsample only along the spatial axes
            data = (
                data.repeat(factor, axis=channel_offset)
                .repeat(factor, axis=channel_offset + 1)
                .repeat(factor, axis=channel_offset + 2)
            )
        elif rescale_factor < 1:
            # Create block_size that leaves the channel dimension unchanged and scales the spatial ones
            block_size = (1,) * channel_offset + (int(1 / rescale_factor),) * 3
            data = block_reduce(data, block_size=block_size, func=np.median)

        data = data[slices]

    if swap_axes:
        data = np.swapaxes(data, 0 + channel_offset, 2 + channel_offset)

    return data


class ImageDataInterface:
    def __init__(
        self,
        dataset_path,
        mode="r",
        output_voxel_size=None,
        custom_fill_value=None,
        concurrency_limit=1,
    ):
        dataset_path = str(Path(dataset_path).resolve())
        self.path = dataset_path
        filename, dataset = split_dataset_path(dataset_path)
        self.ds = open_ds(filename, dataset, mode=mode)
        self.filetype = (
            "zarr" if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5") else "n5"
        )
        self.swap_axes = self.filetype == "n5"
        self.ts = None
        self.voxel_size = self.ds.voxel_size
        self.chunk_shape = self.ds.chunk_shape
        self.roi = self.ds.roi
        self.offset = self.ds.roi.offset

        if "voxel_size" in self.ds.data.attrs:
            self.voxel_size = Coordinate(self.ds.data.attrs["voxel_size"])
            self.roi *= self.voxel_size
            self.offset *= self.voxel_size

        self.custom_fill_value = custom_fill_value
        self.concurrency_limit = concurrency_limit
        if output_voxel_size is not None:
            self.output_voxel_size = output_voxel_size
        else:
            self.output_voxel_size = self.voxel_size

    def to_ndarray_ts(self, roi=None):
        if not self.ts:
            self.ts = open_ds_tensorstore(
                self.path, concurrency_limit=self.concurrency_limit
            )
            self.domain = self.ts.domain
        res = to_ndarray_tensorstore(
            self.ts,
            roi,
            self.voxel_size,
            self.offset,
            self.output_voxel_size,
            self.swap_axes,
            self.custom_fill_value,
        )
        self.ts = None
        return res

    def to_ndarray_ds(self, roi=None):
        return self.ds.to_ndarray(roi)


# idi = ImageDataInterface(
#     "/nrs/cellmap/ackermand/predictions/cellmap_experiments/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.zarr/predictions/2025-02-15_3m/plasmodesmata_affs_lsds/0__affs"
# )
# 1402, 3818, 1336, 176, 101, 510
# 56816, 41197, 2076, 176, 101, 510
# roi = Roi(np.array([2076, 41197, 56816]), 3 * [8 * 216])


# from scipy.ndimage import measurements, gaussian_filter
# import mwatershed as mws
# import fastremap


# def filter_fragments(
#     affs_data: np.ndarray, fragments_data: np.ndarray, filter_val: float
# ) -> None:
#     """Allows filtering of MWS fragments based on mean value of affinities & fragments. Will filter and update the fragment array in-place.

#     Args:
#         aff_data (``np.ndarray``):
#             An array containing affinity data.

#         fragments_data (``np.ndarray``):
#             An array containing fragment data.

#         filter_val (``float``):
#             Threshold to filter if the average value falls below.
#     """

#     average_affs: float = np.mean(affs_data.data, axis=0)

#     filtered_fragments: list = []

#     fragment_ids: np.ndarray = np.unique(fragments_data)

#     for fragment, mean in zip(
#         fragment_ids, measurements.mean(average_affs, fragments_data, fragment_ids)
#     ):
#         if mean < filter_val:
#             filtered_fragments.append(fragment)

#     filtered_fragments: np.ndarray = np.array(
#         filtered_fragments, dtype=fragments_data.dtype
#     )
#     # replace: np.ndarray = np.zeros_like(filtered_fragments)
#     fastremap.mask(fragments_data, filtered_fragments, in_place=True)


# neighborhood = [
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [3, 0, 0],
#     [0, 3, 0],
#     [0, 0, 3],
#     [9, 0, 0],
#     [0, 9, 0],
#     [0, 0, 9],
# ]
# # neighborhood = [[0,0,1],[0,1,0],[1,0,0],[0,0,3],[0,3,0],[3,0,0],[0,0,9],[0,9,0],[9,0,0]]
# import time


# def mutex_watershed_blockwise(
#     data, adjacent_edge_bias=-0.4, lr_bias_ratio=-0.08, filter_val=0.5
# ):
#     if data.dtype == np.uint8:
#         logger.info("Assuming affinities are in [0,255]")
#         max_affinity_value: float = 255.0
#         data = data.astype(np.float64)
#     else:
#         data = data.astype(np.float64)
#         max_affinity_value: float = 1.0

#     data /= max_affinity_value

#     if data.max() < 1e-3:
#         segmentation = np.zeros(data.shape, dtype=np.uint64)
#         return data

#     t0 = time.time()
#     random_noise = np.random.randn(*data.shape) * 0.0001
#     smoothed_affs = (
#         gaussian_filter(data, sigma=(0, *(np.amax(neighborhood, axis=0) / 3))) - 0.5
#     ) * 0.001
#     shift: np.ndarray = np.array(
#         [
#             (
#                 adjacent_edge_bias
#                 if max(offset) <= 1
#                 else np.linalg.norm(offset) * lr_bias_ratio
#             )
#             for offset in neighborhood
#         ]
#     ).reshape((-1, *((1,) * (len(data.shape) - 1))))

#     # raise Exception(data.max(), data.min(), self.neighborhood)

#     # segmentation = mws.agglom(
#     #     data.astype(np.float64) - self.bias,
#     #     self.neighborhood,
#     # )

#     # filter fragments
#     t1 = time.time()
#     segmentation = mws.agglom(
#         data + shift + random_noise + smoothed_affs,
#         offsets=neighborhood,
#     )
#     t2 = time.time()

#     if filter_val > 0.0:
#         filter_fragments(data, segmentation, filter_val)
#     t3 = time.time()
#     # fragment_ids = fastremap.unique(segmentation[segmentation > 0])
#     # fastremap.mask_except(segmentation, filtered_fragments, in_place=True)
#     fastremap.renumber(segmentation, in_place=True)
#     t4 = time.time()
#     print(f"{t1-t0},{t2-t1}, {t3-t2}, {t4-t3}")
#     return segmentation


# import cc3d


# def instancewise_instance_segmentation(segmentation):
#     ids = fastremap.unique(segmentation[segmentation > 0])
#     output = np.zeros_like(segmentation, dtype=np.uint64)
#     for id in ids:
#         cc = cc3d.connected_components(
#             segmentation == id,
#             connectivity=6,
#             binary_image=True,
#         )
#         cc[cc > 0] += np.max(output)
#         output += cc
#     return output


# res = idi.to_ndarray_ts(roi)
# output = mutex_watershed_blockwise(res)
# # %%
# np.unique(output)
# import matplotlib.pyplot as plt
# from cellmap_analyze.util.neuroglancer_util import view_in_neuroglancer

# output = instancewise_instance_segmentation(output)

# view_in_neuroglancer(output=output.astype(np.uint16))
# plt.imshow(output[100, :, :])
# # %%


# # %%

# %%

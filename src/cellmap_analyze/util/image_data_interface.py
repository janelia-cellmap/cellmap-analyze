import logging
import tensorstore as ts
import numpy as np
from funlib.geometry import Coordinate
from funlib.geometry import Roi

from cellmap_analyze.util.io_util import split_dataset_path
from funlib.persistence import open_ds

# Much below taken from flyemflows: https://github.com/janelia-flyem/flyemflows/blob/master/flyemflows/util/util.py
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def open_ds_tensorstore(dataset_path: str, mode="r"):
    # open with zarr or n5 depending on extension
    filetype = (
        "zarr" if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5") else "n5"
    )
    spec = {
        "driver": filetype,
        # "context": {
        #     "data_copy_concurrency": {"limit": 1},
        #     "file_io_concurrency": {"limit": 1},
        # },
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
):
    """Read a region of a tensorstore dataset and return it as a numpy array

    Args:
        dataset ('tensorstore.dataset'): Tensorstore dataset
        roi ('funlib.geometry.Roi'): Region of interest to read

    Returns:
        Numpy array of the region
    """
    if swap_axes:
        print("Swapping axes")
        if roi:
            roi = Roi(roi.begin[::-1], roi.shape[::-1])
        if offset:
            offset = Coordinate(offset[::-1])

    if roi is None:
        return dataset.read().result()

    if offset is None:
        offset = Coordinate(np.zeros(roi.dims, dtype=int))

    if output_voxel_size is None:
        output_voxel_size = voxel_size

    rescale_factor = 1
    if voxel_size != output_voxel_size:
        # in the case where there is a mismatch in voxel sizes, we may need to extra pad to ensure that the output is a multiple of the output voxel size
        original_roi = roi
        roi = original_roi.snap_to_grid(voxel_size)
        rescale_factor = voxel_size[0] / output_voxel_size[0]
        snapped_offset = (original_roi.begin - roi.begin) / output_voxel_size
        snapped_end = (original_roi.end - roi.begin) / output_voxel_size
        snapped_slices = tuple(
            slice(snapped_offset[i], snapped_end[i]) for i in range(3)
        )

    roi -= offset
    roi /= voxel_size

    # Specify the range
    roi_slices = roi.to_slices()

    domain = dataset.domain
    # Compute the valid range
    valid_slices = tuple(
        slice(max(s.start, inclusive_min), min(s.stop, exclusive_max))
        for s, inclusive_min, exclusive_max in zip(
            roi_slices, domain.inclusive_min, domain.exclusive_max
        )
    )

    # Create an array to hold the requested data, filled with a default value (e.g., zeros)
    output_shape = [s.stop - s.start for s in roi_slices]

    if not dataset.fill_value:
        fill_value = 0
    padded_data = np.ones(output_shape, dtype=dataset.dtype.numpy_dtype) * fill_value
    padded_slices = tuple(
        slice(valid_slice.start - s.start, valid_slice.stop - s.start)
        for s, valid_slice in zip(roi_slices, valid_slices)
    )

    # Read the region of interest from the dataset
    padded_data[padded_slices] = dataset[valid_slices].read().result()

    if rescale_factor > 1:
        rescale_factor = voxel_size[0] / output_voxel_size[0]
        padded_data = (
            padded_data.repeat(rescale_factor, 0)
            .repeat(rescale_factor, 1)
            .repeat(rescale_factor, 2)
        )
        padded_data = padded_data[snapped_slices]

    if swap_axes:
        padded_data = np.swapaxes(padded_data, 0, 2)

    return padded_data


class ImageDataInterface:
    def __init__(self, dataset_path, mode="r", output_voxel_size=None):
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
        if output_voxel_size is not None:
            self.output_voxel_size = output_voxel_size
        else:
            self.output_voxel_size = self.voxel_size

    def to_ndarray_ts(self, roi=None):
        if not self.ts:
            self.ts = open_ds_tensorstore(self.path)

        return to_ndarray_tensorstore(
            self.ts,
            roi,
            self.voxel_size,
            self.offset,
            self.output_voxel_size,
            self.swap_axes,
        )

    def to_ndarray_ds(self, roi=None):
        return self.ds.to_ndarray(roi)

# %%
import logging
from pathlib import Path
import tensorstore as ts
import numpy as np
from funlib.geometry import Coordinate
from funlib.geometry import Roi

from cellmap_analyze.util.io_util import split_dataset_path, print_with_datetime
from cellmap_analyze.util.voxel_size_utils import (
    read_raw_voxel_size,
    read_raw_offset,
    scale_voxel_size_to_integers,
)
from funlib.persistence import open_ds
from scipy.ndimage import zoom, map_coordinates
import time
import random

# Much below taken from flyemflows: https://github.com/janelia-flyem/flyemflows/blob/master/flyemflows/util/util.py
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def is_close_to_integer(value, tolerance=0.01):
    """Check if a value is close to an integer within tolerance.

    Args:
        value: Numeric value to check
        tolerance: Maximum difference from nearest integer (default: 0.01 = 1%)

    Returns:
        True if value is within tolerance of an integer
    """
    return abs(value - round(value)) < tolerance


def requires_interpolation(rescale_factors, tolerance=0.01):
    """Check if any rescale factor requires interpolation (non-integer).

    Args:
        rescale_factors: Tuple of per-axis rescale factors
        tolerance: Tolerance for integer check (default: 0.01 = 1%)

    Returns:
        True if any factor is not close to an integer
    """
    return any(not is_close_to_integer(rf, tolerance) for rf in rescale_factors)


def read_with_retries(dataset, valid_slices, max_retries=10, timeout=5, base_delay=1):
    """
    Attempt to read from TensorStore up to `max_retries` times on TimeoutError.

    Parameters
    ----------
    dataset : tensorstore.TensorStore
        Opened TensorStore handle.
    valid_slices : tuple of slice
        The slices you want to read.
    max_retries : int, optional
        How many times to retry on TimeoutError (default: 3).
    timeout : float, optional
        Seconds to wait on each .result(timeout=…) call (default: 30).

           Returns
    -------
    numpy.ndarray
        The result of the successful read.

    Raises
    ------
    TimeoutError
        If all attempts time out.
    """

    for attempt in range(1, max_retries + 1):
        try:
            return dataset[valid_slices].read().result(timeout=timeout * attempt)
        except TimeoutError as e:
            print_with_datetime(
                f"[Attempt {attempt}/{max_retries}] "
                f"Timeout reading {dataset} slices={valid_slices!r}: {e}",
                logger,
                log_type="error",
            )
            if attempt == max_retries:
                # re-raise the last timeout
                raise
            # exponential backoff with jitter
            delay = base_delay * (1.3 ** (attempt - 1))
            jitter = random.uniform(0, base_delay)
            time.sleep(delay + jitter)


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
    max_retries=10,
    timeout=5,
    interpolation_order=0,
):
    """Read a region of a tensorstore dataset and return it as a numpy array

    Args:
        dataset ('tensorstore.dataset'): Tensorstore dataset
        roi ('funlib.geometry.Roi'): Region of interest to read
        voxel_size: Native voxel size of the dataset
        offset: Spatial offset of the dataset
        output_voxel_size: Target voxel size for resampling (if different from native)
        swap_axes: Whether to swap axes for n5 format
        custom_fill_value: Fill value for padding (default: dataset fill_value)
        max_retries: Maximum retry attempts for reading
        timeout: Timeout in seconds for each read attempt
        interpolation_order: Order of interpolation for non-integer resampling
            0 = nearest-neighbor (preserves labels, default)
            1 = linear interpolation (for continuous data)

    Returns:
        Numpy array of the region, resampled to output_voxel_size if specified

    Note:
        For non-integer scale factors (e.g., 8nm→5nm = 1.6x), uses scipy.ndimage.zoom.
        For integer or close-to-integer factors, uses fast paths (repeat/slicing).
        Padding is preserved in physical space regardless of resampling method.
    """

    if output_voxel_size is None:
        output_voxel_size = voxel_size

    # Calculate per-axis rescale factors for anisotropic data
    rescale_factors = tuple(vs / ovs for vs, ovs in zip(voxel_size, output_voxel_size))

    # Check if any axis needs rescaling
    needs_rescaling = any(rf != 1 for rf in rescale_factors)
    # For fast-path direction (up vs down), use first axis as representative
    rescale_factor = rescale_factors[0]

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
        # with ts.Transaction() as txn:
        data = dataset.read().result()
        # Check if any rescaling is needed (anisotropic or isotropic)
        if needs_rescaling:
            # Check if any rescale factor requires interpolation (non-integer)
            if requires_interpolation(rescale_factors):
                # Use scipy.ndimage.zoom for non-integer scaling
                zoom_factors = (1,) * channel_offset + rescale_factors
                data = zoom(data, zoom=zoom_factors, order=interpolation_order)
            else:
                # Use existing fast paths for integer or close-to-integer scaling
                all_up = all(rf >= 1 for rf in rescale_factors)
                all_down = all(rf <= 1 for rf in rescale_factors)
                if all_up:
                    # Apply per-axis upsampling
                    data = (
                        data.repeat(int(round(rescale_factors[0])), axis=0 + channel_offset)
                        .repeat(int(round(rescale_factors[1])), axis=1 + channel_offset)
                        .repeat(int(round(rescale_factors[2])), axis=2 + channel_offset)
                    )
                elif all_down:
                    # Use simple slicing for integer downsampling
                    downsample_factors = tuple(int(round(1 / rf)) for rf in rescale_factors)
                    downsample_slices = (
                        (slice(None),) * channel_offset +
                        tuple(slice(None, None, factor) for factor in downsample_factors)
                    )
                    data = data[downsample_slices]
                else:
                    # Mixed up/down per axis — use zoom for correctness
                    zoom_factors = (1,) * channel_offset + rescale_factors
                    data = zoom(data, zoom=zoom_factors, order=interpolation_order)
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

    # Check if the ROI has no overlap with the dataset
    no_overlap = any(vs.start > vs.stop for vs in valid_slices)

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

    if no_overlap:
        if needs_rescaling:
            # Use output voxel space for the shape
            output_shape = (
                ([dataset.shape[0]] if channel_offset > 0 else [])
                + [int(round(original_roi.shape[i] / output_voxel_size[i])) for i in range(3)]
            )
        else:
            output_shape = (
                ([dataset.shape[0]] if channel_offset > 0 else [])
                + [s.stop - s.start for s in roi_slices]
            )
        fv = 0 if fill_value == "edge" else fill_value
        return np.full(output_shape, fv, dtype=dataset.dtype.numpy_dtype)

    # with ts.Transaction() as txn:
    try:
        data = read_with_retries(dataset, valid_slices, max_retries, timeout)
    except TimeoutError:
        logger.error(
            f"Timeout while reading dataset {dataset} with slices {valid_slices}"
        )
        raise TimeoutError(
            f"Failed to read dataset {dataset} with slices {valid_slices} after {max_retries} retries."
        )
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

    # Resample if needed
    if needs_rescaling:
        # Check if any rescale factor requires interpolation (non-integer)
        if requires_interpolation(rescale_factors):
            # Use map_coordinates for non-integer scaling to ensure global grid alignment
            # Calculate global output positions aligned to dataset origin
            # original_roi is in physical coordinates aligned to output voxel size

            # Number of output voxels needed
            num_output_voxels = tuple(
                int(round(original_roi.shape[i] / output_voxel_size[i]))
                for i in range(3)
            )

            # Global output positions in physical space (relative to dataset origin + offset)
            output_positions = [
                original_roi.begin[i] - offset[i] + np.arange(num_output_voxels[i]) * output_voxel_size[i]
                for i in range(3)
            ]

            # Map output positions to input array coordinates
            # data starts at position (roi.begin * voxel_size + offset) in physical space
            # After subtracting offset and dividing by voxel_size, data starts at roi.begin in voxel coords
            # data array index 0 corresponds to physical position (roi.begin * voxel_size + offset)
            data_start_physical = roi.begin * voxel_size

            # Convert output positions to input array indices
            input_coords = [
                (output_positions[i] - data_start_physical[i]) / voxel_size[i]
                for i in range(3)
            ]

            # Create mesh grid for 3D sampling
            coords_mesh = np.meshgrid(*input_coords, indexing='ij')
            coords_array = np.array([c.ravel() for c in coords_mesh])

            # Sample at exact global grid positions using map_coordinates
            # Handle channels if present
            if channel_offset > 0:
                # Resample each channel separately
                output_data = np.zeros((data.shape[0],) + num_output_voxels, dtype=data.dtype)
                for ch in range(data.shape[0]):
                    resampled = map_coordinates(
                        data[ch],
                        coords_array,
                        order=interpolation_order,
                        mode='constant',
                        cval=0
                    )
                    output_data[ch] = resampled.reshape(num_output_voxels)
                data = output_data
            else:
                data = map_coordinates(
                    data,
                    coords_array,
                    order=interpolation_order,
                    mode='constant',
                    cval=0
                )
                data = data.reshape(num_output_voxels)
        else:
            # Use existing fast paths for integer or close-to-integer scaling
            slices = (slice(None),) * channel_offset + snapped_slices

            all_up = all(rf >= 1 for rf in rescale_factors)
            all_down = all(rf <= 1 for rf in rescale_factors)
            if all_up:
                # Apply per-axis upsampling for anisotropic data
                # Use round() instead of int() to handle factors like 1.999
                data = (
                    data.repeat(int(round(rescale_factors[0])), axis=channel_offset)
                    .repeat(int(round(rescale_factors[1])), axis=channel_offset + 1)
                    .repeat(int(round(rescale_factors[2])), axis=channel_offset + 2)
                )
            elif all_down:
                # Use simple slicing for integer downsampling (preserves exact labels)
                # Calculate per-axis downsampling factors
                downsample_factors = tuple(int(round(1 / rf)) for rf in rescale_factors)

                # Build slicing tuple for downsampling
                # For channels: slice(None), for spatial dims: slice(None, None, factor)
                downsample_slices = (
                    (slice(None),) * channel_offset +
                    tuple(slice(None, None, factor) for factor in downsample_factors)
                )
                data = data[downsample_slices]
            else:
                # Mixed up/down per axis — use zoom for correctness
                zoom_factors = (1,) * channel_offset + rescale_factors
                data = zoom(data, zoom=zoom_factors, order=interpolation_order)

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
        chunk_shape=None,
        max_retries=10,
        timeout=5,
        interpolation_order=0,
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
        self.dtype = self.ds.dtype
        self.chunk_shape = self.ds.chunk_shape
        if chunk_shape is not None:
            if type(chunk_shape) != Coordinate:
                chunk_shape = Coordinate(chunk_shape)
            self.chunk_shape = chunk_shape

        # Read the raw float voxel_size and offset from zarr attributes
        # BEFORE funlib.geometry.Coordinate truncates them to integers.
        raw_voxel_size = read_raw_voxel_size(self.ds)
        raw_offset = read_raw_offset(self.ds)

        # Handle multichannel metadata: strip extra (non-spatial) dims
        n_spatial = 3
        if len(raw_voxel_size) > n_spatial:
            n_extra = len(raw_voxel_size) - n_spatial
            raw_voxel_size = raw_voxel_size[n_extra:]
            raw_offset = raw_offset[n_extra:]

        # Scale float voxel sizes to integers for funlib compatibility
        scaled_vs, scale_factor = scale_voxel_size_to_integers(raw_voxel_size)
        self.original_voxel_size = raw_voxel_size
        self._raw_offset = raw_offset
        self.voxel_size_scale_factor = scale_factor
        self.voxel_size = Coordinate(scaled_vs)

        # Recompute ROI in scaled physical coordinates from the array shape
        array_shape = self.ds.data.shape[-n_spatial:]
        scaled_offset = Coordinate(
            int(round(o * scale_factor)) for o in raw_offset
        )
        self.roi = Roi(scaled_offset, Coordinate(array_shape) * self.voxel_size)
        self.offset = self.roi.offset

        self.custom_fill_value = custom_fill_value
        self.concurrency_limit = concurrency_limit
        if output_voxel_size is not None:
            self.output_voxel_size = output_voxel_size
        else:
            self.output_voxel_size = self.voxel_size

        self.max_retries = max_retries
        self.timeout = timeout
        self.interpolation_order = interpolation_order

    def rescale_to_factor(self, new_scale_factor):
        """Rescale this IDI's internal coordinates to use a new scale factor.

        Used when combining multiple IDIs that need to share the same
        scaled coordinate space (e.g., contact sites).

        Args:
            new_scale_factor: The new integer scale factor to use.
        """
        if new_scale_factor == self.voxel_size_scale_factor:
            return

        self.voxel_size_scale_factor = new_scale_factor
        scaled_vs = tuple(
            int(round(v * new_scale_factor)) for v in self.original_voxel_size
        )
        self.voxel_size = Coordinate(scaled_vs)

        array_shape = self.ds.data.shape[-3:]
        scaled_offset = Coordinate(
            int(round(o * new_scale_factor)) for o in self._raw_offset
        )
        self.roi = Roi(scaled_offset, Coordinate(array_shape) * self.voxel_size)
        self.offset = self.roi.offset
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
            self.max_retries,
            self.timeout,
            self.interpolation_order,
        )
        self.ts = None
        return res

    def to_ndarray_ds(self, roi=None):
        return self.ds.to_ndarray(roi)

# %%
import logging
from pathlib import Path
import numpy as np
import xarray as xr
import zarr
import tensorstore as ts

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class Coordinate(tuple):
    """Lightweight replacement for funlib.geometry.Coordinate.

    Supports element-wise arithmetic operations with other Coordinates
    and scalar values.
    """

    def __new__(cls, values):
        if isinstance(values, Coordinate):
            return values
        return super().__new__(cls, tuple(values))

    def __add__(self, other):
        if isinstance(other, (tuple, list, Coordinate)):
            return Coordinate(a + b for a, b in zip(self, other))
        return Coordinate(a + other for a in self)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (tuple, list, Coordinate)):
            return Coordinate(a - b for a, b in zip(self, other))
        return Coordinate(a - other for a in self)

    def __rsub__(self, other):
        if isinstance(other, (tuple, list, Coordinate)):
            return Coordinate(b - a for a, b in zip(self, other))
        return Coordinate(other - a for a in self)

    def __mul__(self, other):
        if isinstance(other, (tuple, list, Coordinate)):
            return Coordinate(a * b for a, b in zip(self, other))
        return Coordinate(a * other for a in self)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (tuple, list, Coordinate)):
            return Coordinate(a / b for a, b in zip(self, other))
        return Coordinate(a / other for a in self)

    def __rtruediv__(self, other):
        if isinstance(other, (tuple, list, Coordinate)):
            return Coordinate(b / a for a, b in zip(self, other))
        return Coordinate(other / a for a in self)

    def __floordiv__(self, other):
        if isinstance(other, (tuple, list, Coordinate)):
            return Coordinate(a // b for a, b in zip(self, other))
        return Coordinate(a // other for a in self)

    def __rfloordiv__(self, other):
        if isinstance(other, (tuple, list, Coordinate)):
            return Coordinate(b // a for a, b in zip(self, other))
        return Coordinate(other // a for a in self)

    def __neg__(self):
        return Coordinate(-a for a in self)

    def __abs__(self):
        return Coordinate(abs(a) for a in self)

    def __eq__(self, other):
        if isinstance(other, (tuple, list, Coordinate)):
            if len(self) != len(other):
                return False
            return all(a == b for a, b in zip(self, other))
        return False

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return f"Coordinate({tuple(self)})"


class Roi:
    """Lightweight replacement for funlib.geometry.Roi.

    Handles regions of interest with begin/shape supporting anisotropic operations.
    """

    def __init__(self, begin, shape):
        self._begin = Coordinate(begin)
        self._shape = Coordinate(shape)

    @property
    def begin(self):
        return self._begin

    @property
    def shape(self):
        return self._shape

    @property
    def end(self):
        return self._begin + self._shape

    @property
    def offset(self):
        return self._begin

    @property
    def dims(self):
        return len(self._begin)

    def get_begin(self):
        return self._begin

    def get_end(self):
        return self.end

    def snap_to_grid(self, grid):
        """Snap ROI boundaries to grid - critical for voxel alignment."""
        grid = Coordinate(grid)
        # Snap begin down to grid
        snapped_begin = Coordinate(
            (b // g) * g for b, g in zip(self._begin, grid)
        )
        # Snap end up to grid
        end = self.end
        snapped_end = Coordinate(
            ((e + g - 1) // g) * g if e % g != 0 else e for e, g in zip(end, grid)
        )
        snapped_shape = snapped_end - snapped_begin
        return Roi(snapped_begin, snapped_shape)

    def grow(self, amount_neg=0, amount_pos=0):
        """Expand ROI by given amounts."""
        if isinstance(amount_neg, (int, float)):
            amount_neg = Coordinate([amount_neg] * self.dims)
        else:
            amount_neg = Coordinate(amount_neg)

        if isinstance(amount_pos, (int, float)):
            amount_pos = Coordinate([amount_pos] * self.dims)
        else:
            amount_pos = Coordinate(amount_pos)

        new_begin = self._begin - amount_neg
        new_shape = self._shape + amount_neg + amount_pos
        return Roi(new_begin, new_shape)

    def intersect(self, other):
        """Return intersection of this ROI with another."""
        new_begin = Coordinate(
            max(a, b) for a, b in zip(self._begin, other.begin)
        )
        new_end = Coordinate(
            min(a, b) for a, b in zip(self.end, other.end)
        )
        new_shape = Coordinate(
            max(0, e - b) for b, e in zip(new_begin, new_end)
        )
        return Roi(new_begin, new_shape)

    def to_slices(self):
        """Convert ROI to tuple of slices."""
        return tuple(
            slice(int(b), int(e)) for b, e in zip(self._begin, self.end)
        )

    def __truediv__(self, other):
        if isinstance(other, (tuple, list, Coordinate)):
            other = Coordinate(other)
            return Roi(self._begin / other, self._shape / other)
        return Roi(self._begin / other, self._shape / other)

    def __mul__(self, other):
        if isinstance(other, (tuple, list, Coordinate)):
            other = Coordinate(other)
            return Roi(self._begin * other, self._shape * other)
        return Roi(self._begin * other, self._shape * other)

    def __sub__(self, other):
        if isinstance(other, (tuple, list, Coordinate)):
            other = Coordinate(other)
            return Roi(self._begin - other, self._shape)
        return Roi(self._begin - other, self._shape)

    def __eq__(self, other):
        if not isinstance(other, Roi):
            return False
        return self._begin == other.begin and self._shape == other.shape

    def __repr__(self):
        return f"Roi(begin={tuple(self._begin)}, shape={tuple(self._shape)})"


def open_ds_tensorstore(dataset_path: str, mode="r", concurrency_limit=None):
    """Open a zarr or n5 dataset with TensorStore."""
    filetype = (
        "zarr" if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5") else "n5"
    )
    spec = {
        "driver": filetype,
        "kvstore": {"driver": "file", "path": dataset_path},
    }
    if concurrency_limit:
        spec["context"] = {
            "data_copy_concurrency": {"limit": concurrency_limit},
            "file_io_concurrency": {"limit": concurrency_limit},
        }

    if mode == "r":
        return ts.open(spec, read=True, write=False).result()
    else:
        return ts.open(spec, read=False, write=True).result()


def split_dataset_path(dataset_path: str):
    """Split a dataset path into filename and dataset components."""
    dataset_path = str(dataset_path)
    # Find the zarr or n5 extension
    zarr_idx = dataset_path.rfind(".zarr")
    n5_idx = dataset_path.rfind(".n5")

    if zarr_idx > n5_idx:
        ext_idx = zarr_idx
        ext_len = 5  # len(".zarr")
    elif n5_idx >= 0:
        ext_idx = n5_idx
        ext_len = 3  # len(".n5")
    else:
        raise ValueError(f"Could not find .zarr or .n5 in path: {dataset_path}")

    filename = dataset_path[: ext_idx + ext_len]
    dataset = dataset_path[ext_idx + ext_len :]
    if dataset.startswith("/"):
        dataset = dataset[1:]

    return filename, dataset


class XarrayImageDataInterface:
    """Image data interface using xarray-tensorstore with anisotropic voxel support.

    Attributes:
        path (str): Full path to dataset
        voxel_size (Coordinate): Per-axis voxel sizes in physical units
        dtype: NumPy dtype of the data
        chunk_shape (Coordinate): Chunk sizes per axis
        roi (Roi): Full region of interest in physical coordinates
        offset (Coordinate): Dataset offset in physical coordinates
        output_voxel_size (Coordinate): Target voxel size for resampling
    """

    def __init__(
        self,
        dataset_path: str,
        mode: str = "r",
        output_voxel_size=None,
        custom_fill_value=None,
        concurrency_limit: int = None,
        chunk_shape=None,
    ):
        dataset_path = str(Path(dataset_path).resolve())
        self.path = dataset_path
        self.mode = mode
        self.custom_fill_value = custom_fill_value
        self.concurrency_limit = concurrency_limit

        # Determine file type
        self.filetype = (
            "zarr" if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5") else "n5"
        )

        # Extract metadata from zarr/n5
        self._extract_metadata(dataset_path)

        # Override chunk_shape if provided
        if chunk_shape is not None:
            if not isinstance(chunk_shape, Coordinate):
                chunk_shape = Coordinate(chunk_shape)
            self.chunk_shape = chunk_shape

        # Set output voxel size
        if output_voxel_size is not None:
            if not isinstance(output_voxel_size, Coordinate):
                output_voxel_size = Coordinate(output_voxel_size)
            self.output_voxel_size = output_voxel_size
        else:
            self.output_voxel_size = self.voxel_size

        # Lazy-loaded xarray DataArray
        self._data_array = None
        self._ds = None  # For backward compatibility with funlib.persistence

    def _extract_metadata(self, dataset_path: str):
        """Extract voxel_size, offset, shape, dtype, chunk_shape from zarr/n5 metadata."""
        filename, dataset = split_dataset_path(dataset_path)

        if self.filetype == "zarr":
            root_store = zarr.open(filename, mode="r")
        else:
            root_store = zarr.open(filename, mode="r")

        # Navigate to the dataset
        store = root_store
        if dataset:
            for part in dataset.split("/"):
                if part:
                    store = store[part]

        # Get array properties
        self.dtype = store.dtype
        array_shape = store.shape
        self.chunk_shape = Coordinate(store.chunks)

        # Handle channel dimension (if 4D array)
        self._channel_offset = 0
        if len(array_shape) == 4:
            self._channel_offset = 1
            spatial_shape = array_shape[1:]
        else:
            spatial_shape = array_shape

        # Try to get voxel_size and offset from attributes
        attrs = dict(store.attrs)

        # First check direct attributes
        voxel_size = None
        offset = None

        if "voxel_size" in attrs:
            voxel_size = Coordinate(attrs["voxel_size"])
        elif "resolution" in attrs:
            voxel_size = Coordinate(attrs["resolution"])
        elif "pixelResolution" in attrs:
            # N5 style
            pixel_res = attrs["pixelResolution"]
            if isinstance(pixel_res, dict):
                voxel_size = Coordinate(pixel_res.get("dimensions", [1, 1, 1]))
            else:
                voxel_size = Coordinate(pixel_res)

        if "offset" in attrs:
            offset = Coordinate(attrs["offset"])
        elif "translation" in attrs:
            offset = Coordinate(attrs["translation"])

        # If not found, check parent for OME-Zarr multiscales metadata
        if voxel_size is None or offset is None:
            voxel_size_from_multiscales, offset_from_multiscales = (
                self._extract_from_multiscales(root_store, dataset)
            )
            if voxel_size is None and voxel_size_from_multiscales is not None:
                voxel_size = voxel_size_from_multiscales
            if offset is None and offset_from_multiscales is not None:
                offset = offset_from_multiscales

        # Set defaults if still not found
        if voxel_size is None:
            voxel_size = Coordinate([1] * len(spatial_shape))
        if offset is None:
            offset = Coordinate([0] * len(spatial_shape))

        self.voxel_size = voxel_size
        self.offset = offset

        # Calculate ROI
        roi_shape = Coordinate(
            s * v for s, v in zip(spatial_shape, self.voxel_size)
        )
        self.roi = Roi(self.offset, roi_shape)

    def _extract_from_multiscales(self, root_store, dataset: str):
        """Extract voxel_size and offset from OME-Zarr multiscales metadata."""
        if not dataset:
            return None, None

        # Get the array name (last part of path) and parent path
        parts = [p for p in dataset.split("/") if p]
        if not parts:
            return None, None

        array_name = parts[-1]  # e.g., "s0"
        parent_parts = parts[:-1]  # e.g., ["recon-1", "em", "fibsem-uint8"]

        # Navigate to parent group
        parent = root_store
        for part in parent_parts:
            try:
                parent = parent[part]
            except KeyError:
                return None, None

        # Check for multiscales attribute
        parent_attrs = dict(parent.attrs)
        if "multiscales" not in parent_attrs:
            return None, None

        multiscales = parent_attrs["multiscales"]
        if not isinstance(multiscales, list) or len(multiscales) == 0:
            return None, None

        # Find our dataset in the multiscales
        for ms in multiscales:
            datasets = ms.get("datasets", [])
            for ds in datasets:
                if ds.get("path") == array_name:
                    # Found our dataset - extract coordinateTransformations
                    transforms = ds.get("coordinateTransformations", [])
                    voxel_size = None
                    offset = None

                    for transform in transforms:
                        if transform.get("type") == "scale":
                            voxel_size = Coordinate(transform.get("scale", [1, 1, 1]))
                        elif transform.get("type") == "translation":
                            offset = Coordinate(transform.get("translation", [0, 0, 0]))

                    return voxel_size, offset

        return None, None

    def _get_data_array(self) -> xr.DataArray:
        """Get or create the xarray DataArray backed by tensorstore."""
        if self._data_array is not None:
            return self._data_array

        from xarray_tensorstore import _TensorStoreAdapter

        ts_arr = self._get_tensorstore()

        # Build coordinate arrays from voxel_size and offset
        array_shape = ts_arr.shape
        if self._channel_offset:
            spatial_shape = array_shape[1:]
        else:
            spatial_shape = array_shape

        coords = {}
        dim_names = ["z", "y", "x"]
        if self._channel_offset:
            dim_names = ["c"] + dim_names
            coords["c"] = np.arange(array_shape[0])

        for i, (name, size) in enumerate(zip(dim_names[-3:], spatial_shape)):
            coords[name] = (
                np.arange(size) * self.voxel_size[i] + self.offset[i]
            )

        adapter = _TensorStoreAdapter(ts_arr)
        self._data_array = xr.DataArray(
            xr.Variable(dim_names if self._channel_offset else dim_names[-3:], adapter),
            dims=dim_names if self._channel_offset else dim_names[-3:],
            coords=coords,
        )
        return self._data_array

    def _get_tensorstore(self):
        """Get the tensorstore array for direct access (cached)."""
        if not hasattr(self, '_ts_arr') or self._ts_arr is None:
            self._ts_arr = open_ds_tensorstore(
                self.path, mode=self.mode, concurrency_limit=self.concurrency_limit
            )
        return self._ts_arr

    def _read_xarray(self, data: xr.DataArray) -> np.ndarray:
        """Read an xarray DataArray via xarray-tensorstore."""
        return data.values

    def to_ndarray_ts(self, roi: Roi = None) -> np.ndarray:
        """Read region as numpy array with per-axis resampling.

        Args:
            roi: Region of interest in physical coordinates.
                 If None, reads entire dataset.

        Returns:
            NumPy array at output_voxel_size resolution
        """
        da = self._get_data_array()
        needs_resampling = self.voxel_size != self.output_voxel_size

        if roi is None:
            data = da
            needs_padding = False
        else:
            spatial_dims = ["z", "y", "x"]
            data_roi = self.roi
            intersection = roi.intersect(data_roi)

            # Calculate padding needed (in output voxels)
            pad_before = [
                max(0, int((data_roi.begin[i] - roi.begin[i]) / self.output_voxel_size[i]))
                for i in range(3)
            ]
            pad_after = [
                max(0, int((roi.end[i] - data_roi.end[i]) / self.output_voxel_size[i]))
                for i in range(3)
            ]
            needs_padding = any(p > 0 for p in pad_before + pad_after)

            if intersection.shape != Coordinate([0, 0, 0]):
                isel_dict = {}
                for i, dim in enumerate(spatial_dims):
                    start_idx = int((intersection.begin[i] - self.offset[i]) / self.voxel_size[i])
                    end_idx = int((intersection.end[i] - self.offset[i]) / self.voxel_size[i])
                    isel_dict[dim] = slice(start_idx, end_idx)
                data = da.isel(**isel_dict)
            else:
                # ROI is completely outside data bounds
                output_shape = tuple(
                    int(roi.shape[i] / self.output_voxel_size[i]) for i in range(3)
                )
                if self._channel_offset:
                    output_shape = (da.shape[0],) + output_shape
                fill_value = self.custom_fill_value if self.custom_fill_value else 0
                return np.full(output_shape, fill_value, dtype=self.dtype)

        if needs_resampling:
            data = self._resample(data, roi)

        result = self._read_xarray(data)

        # Apply padding if needed
        if roi is not None and needs_padding:
            fill_value = self.custom_fill_value if self.custom_fill_value else 0
            if self._channel_offset:
                pad_width = [(0, 0)] + list(zip(pad_before, pad_after))
            else:
                pad_width = list(zip(pad_before, pad_after))

            if fill_value == "edge":
                result = np.pad(result, pad_width, mode="edge")
            else:
                result = np.pad(result, pad_width, mode="constant", constant_values=fill_value)

        return result

    def _resample(self, data: xr.DataArray, roi: Roi = None) -> xr.DataArray:
        """Resample data to output_voxel_size using xarray methods.

        Uses interp() for upsampling and coarsen() for downsampling.
        Supports arbitrary per-axis resampling factors (e.g., Z by 2.5x, Y by 1.3x, X unchanged).
        """
        spatial_dims = ["z", "y", "x"]

        # Calculate per-axis rescale factors (source_voxel / output_voxel)
        # factor > 1 means upsampling, factor < 1 means downsampling
        rescale_factors = tuple(
            self.voxel_size[i] / self.output_voxel_size[i] for i in range(3)
        )

        # Check if upsampling or downsampling per axis
        is_upsampling = any(f > 1 for f in rescale_factors)
        is_downsampling = any(f < 1 for f in rescale_factors)

        if is_upsampling and is_downsampling:
            raise ValueError(
                f"Mixed upsampling/downsampling not supported. "
                f"rescale_factors={rescale_factors}"
            )

        if is_upsampling:
            # Use interp() for upsampling - supports arbitrary factors per axis
            new_coords = {}
            for i, dim in enumerate(spatial_dims):
                factor = rescale_factors[i]
                if factor > 1 and dim in data.coords:
                    old_coords = data.coords[dim].values
                    if len(old_coords) > 0:
                        # Generate new coordinates at output voxel spacing
                        # Start from first coordinate, step by output_voxel_size
                        start = old_coords[0]
                        # End coordinate is the last original coord
                        # Number of new points = original_extent / output_voxel_size
                        num_new_points = int(len(old_coords) * factor)
                        new_coords[dim] = np.arange(num_new_points) * self.output_voxel_size[i] + start

            if new_coords:
                data = data.interp(**new_coords, method="nearest")

        elif is_downsampling:
            # Use coarsen() for downsampling - requires integer factors
            coarsen_kwargs = {}
            for i, dim in enumerate(spatial_dims):
                factor = rescale_factors[i]
                if factor < 1:
                    # coarsen factor is the inverse: how many source voxels per output voxel
                    coarsen_factor = int(round(1 / factor))
                    if coarsen_factor > 1:
                        coarsen_kwargs[dim] = coarsen_factor

            if coarsen_kwargs:
                data = data.coarsen(**coarsen_kwargs, boundary="trim").median()

        return data

    def to_ndarray_ds(self, roi: Roi = None) -> np.ndarray:
        """Compatibility method using funlib.persistence.

        This is provided for backward compatibility with code that uses
        the funlib.persistence backend.
        """
        if self._ds is None:
            from funlib.persistence import open_ds
            from cellmap_analyze.util.io_util import split_dataset_path as split_ds_path

            filename, dataset = split_ds_path(self.path)
            self._ds = open_ds(filename, dataset, mode=self.mode)

        return self._ds.to_ndarray(roi)

    @property
    def ds(self):
        """Access to underlying funlib dataset for write operations."""
        if self._ds is None:
            from funlib.persistence import open_ds
            from cellmap_analyze.util.io_util import split_dataset_path as split_ds_path

            filename, dataset = split_ds_path(self.path)
            self._ds = open_ds(filename, dataset, mode=self.mode)

        return self._ds

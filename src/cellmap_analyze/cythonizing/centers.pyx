# centers.pyx
# cython: language_level=3

# ——— Imports & C‐API initialization ———
import numpy as np
cimport numpy as np
import scipy.ndimage
from libc.stdint cimport uint64_t
from libcpp cimport bool
from libcpp.map cimport map as cpp_map

# Initialize the NumPy C-API (required for any cimported np.ndarray)
np.import_array()


# ——— C++ extern declarations ———
cdef extern from "impl/centers.hpp":
    cdef struct Center:
        double z
        double y
        double x
        double sum_r2

    cpp_map[uint64_t, Center] centers(
        size_t size_z,
        size_t size_y,
        size_t size_x,
        const uint64_t* labels,
        bool compute_sum_r2,
        bool center_on_voxels,
        const double* voxel_size,
        const double* offset
    )


# ——— Python-visible functions ———
def find_centers_cpp(np.ndarray[uint64_t, ndim=3] labels not None,
                     bint compute_sum_r2=False,
                     bint center_on_voxels=True,
                     double[:] voxel_size=None,
                     double[:] offset=None):
    """
    Compute connected-component centers via a C++ backend.

    labels:           3D uint64 array of labels; will be made C-contiguous.
    compute_sum_r2:   whether to compute per-label second moment.
    voxel_size:       optional length-3 memoryview with voxel sizes (z, y, x).
    offset:           optional length-3 memoryview of doubles.
    """
    # 1) ensure labels are C-contiguous
    if not labels.flags['C_CONTIGUOUS']:
        labels = np.ascontiguousarray(labels)
    cdef uint64_t* labels_data = <uint64_t*>labels.data

    # 2) prepare voxel_size pointer
    cdef double _voxel_size_arr[3]
    cdef const double* voxel_size_ptr

    if voxel_size is None:
        _voxel_size_arr[0] = 1.0
        _voxel_size_arr[1] = 1.0
        _voxel_size_arr[2] = 1.0
        voxel_size_ptr = _voxel_size_arr
    else:
        if voxel_size.ndim != 1 or voxel_size.shape[0] != 3:
            raise ValueError("`voxel_size` must be a 1D buffer of length 3")
        voxel_size_ptr = &voxel_size[0]

    # 3) prepare offset pointer
    cdef double _offset_arr[3]
    cdef const double* offset_ptr

    if offset is None:
        _offset_arr[0] = 0.0
        _offset_arr[1] = 0.0
        _offset_arr[2] = 0.0
        offset_ptr = _offset_arr
    else:
        if offset.ndim != 1 or offset.shape[0] != 3:
            raise ValueError("`offset` must be a 1D buffer of length 3")
        offset_ptr = &offset[0]

    # 4) call the C++ backend
    cdef cpp_map[uint64_t, Center] result = centers(
        labels.shape[0],
        labels.shape[1],
        labels.shape[2],
        labels_data,
        compute_sum_r2,
        center_on_voxels,
        voxel_size_ptr,
        offset_ptr
    )
    return result


def find_centers_scipy(np.ndarray components, np.ndarray ids):
    """
    Fallback pure-Python/SciPy implementation.
    """
    return np.array(
        scipy.ndimage.measurements.center_of_mass(
            np.ones_like(components),
            components,
            ids
        )
    )


def find_centers(np.ndarray components,
                 np.ndarray ids,
                 compute_sum_r2=False,
                 center_on_voxels=True,
                 voxel_size=None,
                 offset=None):
    """
    Wrapper that dispatches to C++ for 3D data or to SciPy otherwise.

    Args:
        components: Label array
        ids: Array of label IDs to compute centers for
        compute_sum_r2: Whether to compute sum of squared distances
        center_on_voxels: Whether to center on voxel centers (add 0.5)
        voxel_size: Tuple or array of voxel sizes (z, y, x). Can be scalar for isotropic.
        offset: Offset to add to coordinates
    """
    if offset is None:
        offset = np.zeros(3, dtype=np.double)

    # Handle voxel_size - convert scalar or tuple to numpy array
    if voxel_size is None:
        voxel_size_arr = np.ones(3, dtype=np.double)
    elif np.isscalar(voxel_size):
        # Backward compatibility: scalar voxel_edge_length
        voxel_size_arr = np.array([voxel_size, voxel_size, voxel_size], dtype=np.double)
    else:
        voxel_size_arr = np.asarray(voxel_size, dtype=np.double)
        if voxel_size_arr.size != 3:
            raise ValueError(f"voxel_size must have 3 elements, got {voxel_size_arr.size}")
        voxel_size_arr = voxel_size_arr.flatten()

    if components.ndim == 3:
        c_results = find_centers_cpp(
            components.astype(np.uint64),
            compute_sum_r2,
            center_on_voxels,
            voxel_size_arr,
            offset
        )
        if compute_sum_r2:
            coords = np.array([[c_results[i]["z"],
                                 c_results[i]["y"],
                                 c_results[i]["x"]]
                                for i in ids], dtype=np.double)
            sums   = np.array([c_results[i]["sum_r2"] for i in ids],
                              dtype=np.double)
            return coords, sums
        else:
            return np.array([[c_results[i]["z"],
                              c_results[i]["y"],
                              c_results[i]["x"]]
                             for i in ids], dtype=np.double)
    else:
        return find_centers_scipy(components, ids)

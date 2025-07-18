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
        double voxel_edge_length,
        const double* offset
    )


# ——— Python-visible functions ———
def find_centers_cpp(np.ndarray[uint64_t, ndim=3] labels not None,
                     bint compute_sum_r2=False,
                     bint center_on_voxels=True,
                     double voxel_edge_length=1.0,
                     double[:] offset=None):
    """
    Compute connected-component centers via a C++ backend.

    labels:           3D uint64 array of labels; will be made C-contiguous.
    compute_sum_r2:   whether to compute per-label second moment.
    voxel_edge_length: scale factor for converting voxel indices to physical coords.
    offset:           optional length-3 memoryview of doubles.
    """
    # 1) ensure labels are C-contiguous
    if not labels.flags['C_CONTIGUOUS']:
        labels = np.ascontiguousarray(labels)
    cdef uint64_t* labels_data = <uint64_t*>labels.data

    # 2) prepare offset pointer
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

    # 3) call the C++ backend
    cdef cpp_map[uint64_t, Center] result = centers(
        labels.shape[0],
        labels.shape[1],
        labels.shape[2],
        labels_data,
        compute_sum_r2,
        center_on_voxels,
        voxel_edge_length,
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
                 voxel_edge_length=1.0,
                 offset=None):
    """
    Wrapper that dispatches to C++ for 3D data or to SciPy otherwise.
    """
    if offset is None:
        offset = np.zeros(3, dtype=np.double)

    if components.ndim == 3:
        c_results = find_centers_cpp(
            components.astype(np.uint64),
            compute_sum_r2,
            center_on_voxels,
            voxel_edge_length,
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

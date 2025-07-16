# The following code was taken from funlib.evaluate: https://github.com/funkelab/funlib.evaluate/blob/master/funlib/evaluate/centers.pyx
from libc.stdint cimport uint64_t
from libcpp cimport bool
from libcpp.map cimport map as cpp_map
import numpy as np
cimport numpy as np
import scipy.ndimage

def find_centers_cpp(np.ndarray[uint64_t, ndim=3] labels, bool compute_sum_r2=False):

    # the C++ part assumes contiguous memory, make sure we have it (and do 
    # nothing, if we do)
    if not labels.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous label arrray (avoid this by passing C_CONTIGUOUS arrays)")
        labels = np.ascontiguousarray(labels)

    cdef uint64_t* labels_data
    labels_data = <uint64_t*>labels.data

    return centers(
        labels.shape[0],
        labels.shape[1],
        labels.shape[2],
        labels_data,
        compute_sum_r2)

cdef extern from "impl/centers.hpp":

    struct Center:
        double z
        double y
        double x
        double sum_r2  # only if compute_sum_r2 is True

    cpp_map[uint64_t, Center] centers(
            size_t size_z,
            size_t size_y,
            size_t size_x,
            const uint64_t* labels,
            bool compute_sum_r2);


def find_centers_scipy(components, ids):
    return np.array(scipy.ndimage.measurements.center_of_mass(
            np.ones_like(components),
            components,
            ids))


def find_centers(components, ids, compute_sum_r2=False):

    if len(components.shape) == 3:

        centers = find_centers_cpp(components.astype(np.uint64), compute_sum_r2)
        if compute_sum_r2:
            return np.array([
                [centers[i]['z'], centers[i]['y'], centers[i]['x']] for i in ids
            ]), np.array([centers[i]['sum_r2'] for i in ids])
        return np.array([
            [centers[i]['z'], centers[i]['y'], centers[i]['x']]
            for i in ids
        ])

    else:

        return find_centers_scipy(components, ids)
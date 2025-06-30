# src/cellmap_analyze/cythonizing/touching.pyx
# cython: boundscheck=False, wraparound=False, language_level=3
import cython
import numpy as np
cimport numpy as np

#––– Fused types for data vs. object_labels –––
ctypedef fused DTYPE_data:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t

ctypedef fused DTYPE_obj:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t


ctypedef np.uint8_t U8_t  # mask is always uint8

cdef inline int abs_i(int x):
    return x if x >= 0 else -x

def get_touching_ids(
        np.ndarray[DTYPE_data, ndim=3] data not None,
        np.ndarray[U8_t,      ndim=3] mask not None,
        int connectivity = 2,
        np.ndarray[U8_t,      ndim=3] original_data_region_mask = None,
        np.ndarray[DTYPE_obj, ndim=3] object_labels = None):
    """
    Cython‐accelerated 3D touching‐labels.
    `data` and `object_labels` can each be any unsigned‐int array,
    and need not share the same dtype.
    """

    cdef:
        # dimensions
        int D0 = data.shape[0], D1 = data.shape[1], D2 = data.shape[2]
        # loop & neighbor indices
        int z, y, x, dz, dy, dx, nz, ny, nx
        int offsets[26][3]
        int m_off = 0, n_off
        # label values
        DTYPE_data lbl0, lbl1

        # fast memoryviews
        DTYPE_data[:, :, :] dv = data
        U8_t[:,       :, :] mv = mask

        DTYPE_obj[:, :, :] ov
        bint has_obj = object_labels is not None
        
        U8_t[:,       :, :] odrmv
        bint has_odrmv = original_data_region_mask is not None

        # Python‐level result
        object result = set()
        object pair

    if has_obj:
        ov = object_labels

    if has_odrmv:
        odrmv = original_data_region_mask
    
    # Build neighbor‐offset list based on connectivity
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                if connectivity == 1 and (abs_i(dz) + abs_i(dy) + abs_i(dx) != 1):
                    continue
                if connectivity == 2 and (abs_i(dz) + abs_i(dy) + abs_i(dx) > 2):
                    continue
                offsets[m_off][0] = dz
                offsets[m_off][1] = dy
                offsets[m_off][2] = dx
                m_off += 1

    # Main 3D scan
    for z in range(D0):
        for y in range(D1):
            for x in range(D2):
                if not mv[z, y, x]:
                    continue

                lbl0 = dv[z, y, x]

                for n_off in range(m_off):
                    dz = offsets[n_off][0]
                    dy = offsets[n_off][1]
                    dx = offsets[n_off][2]

                    nz = z + dz
                    if nz < 0 or nz >= D0: continue
                    ny = y + dy
                    if ny < 0 or ny >= D1: continue
                    nx = x + dx
                    if nx < 0 or nx >= D2: continue

                    if not mv[nz, ny, nx]:
                        continue
                    
                    if has_odrmv and odrmv[nz, ny, nx] == odrmv[z, y, x]:
                        # only want to keep if one is in the original mask, otherwise we will get self-self contacts which may override blockwise
                        continue

                    lbl1 = dv[nz, ny, nx]
                    if lbl1 == lbl0:
                        continue

                    if has_obj and ov[nz, ny, nx] != ov[z, y, x]:
                        continue

                    # add sorted pair
                    if lbl0 < lbl1:
                        pair = (lbl0, lbl1)
                    else:
                        pair = (lbl1, lbl0)
                    result.add(pair)

    return result

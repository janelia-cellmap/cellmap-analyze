# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, language_level=3
from libc.math cimport sqrt, INFINITY, NAN
from libcpp.pair cimport pair
from libcpp.queue cimport priority_queue

import numpy as np
from numpy cimport npy_intp, npy_uint8, npy_uint16, npy_uint32, npy_uint64

ctypedef fused label_t:
    npy_uint8
    npy_uint16
    npy_uint32
    npy_uint64


def nearest_source_values(
    label_t[:, :, ::1] labels not None,
    sources,
    source_values,
    voxel_size,
):
    """For every labelled voxel, the geodesic distance (26-connected steps of
    their Euclidean length, through voxels of its own label) to the nearest of
    ``sources`` (voxel indices, (n, 3)) of its label, and the mean of
    ``source_values`` over the nearest ones: a voxel takes the mean of the
    values its tied shortest-path predecessors carry. Unreached voxels get an
    infinite distance and a NaN value."""
    cdef:
        npy_intp P = labels.shape[0], R = labels.shape[1], C = labels.shape[2]
        npy_intp sp = R * C, sr = C, n = P * R * C
        npy_intp off[26]
        npy_intp dz_[26]
        npy_intp dy_[26]
        npy_intp dx_[26]
        double weight[26]
        double vz = voxel_size[0], vy = voxel_size[1], vx = voxel_size[2]
        npy_intp k = 0, i, v, u, p, r, c, rem, m
        int dz, dy, dx
        double d, nd, value, tolerance
        priority_queue[pair[double, npy_intp]] queue
        pair[double, npy_intp] top
        label_t* lab = &labels[0, 0, 0]
        label_t L
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                off[k] = dz * sp + dy * sr + dx
                dz_[k] = dz
                dy_[k] = dy
                dx_[k] = dx
                weight[k] = sqrt((dz * vz) ** 2 + (dy * vy) ** 2 + (dx * vx) ** 2)
                k += 1
    tolerance = 1e-9 * min(vz, vy, vx)

    distance_arr = np.full(n, np.inf)
    total_arr = np.zeros(n)
    count_arr = np.zeros(n, dtype=np.uint32)
    done_arr = np.zeros(n, dtype=np.uint8)
    cdef double[::1] distance = distance_arr
    cdef double[::1] total = total_arr
    cdef npy_uint32[::1] count = count_arr
    cdef npy_uint8[::1] done = done_arr
    cdef npy_intp[:, ::1] src = np.ascontiguousarray(sources, dtype=np.intp).reshape(-1, 3)
    cdef double[::1] src_values = np.ascontiguousarray(source_values, dtype=np.float64)

    with nogil:
        for i in range(src.shape[0]):
            v = src[i, 0] * sp + src[i, 1] * sr + src[i, 2]
            if lab[v] == 0:
                continue
            if distance[v] == 0:
                total[v] += src_values[i]
                count[v] += 1
                continue
            distance[v] = 0
            total[v] = src_values[i]
            count[v] = 1
            queue.push(pair[double, npy_intp](0.0, v))

        while not queue.empty():
            top = queue.top()
            queue.pop()
            v = top.second
            if done[v]:
                continue
            done[v] = 1
            d = distance[v]
            value = total[v] / count[v]
            L = lab[v]
            p = v // sp
            rem = v - p * sp
            r = rem // sr
            c = rem - r * sr
            for m in range(26):
                if (p + dz_[m] < 0 or p + dz_[m] >= P or r + dy_[m] < 0 or r + dy_[m] >= R
                        or c + dx_[m] < 0 or c + dx_[m] >= C):
                    continue
                u = v + off[m]
                if lab[u] != L or done[u]:
                    continue
                nd = d + weight[m]
                if nd < distance[u] - tolerance:
                    distance[u] = nd
                    total[u] = value
                    count[u] = 1
                    # the queue is a max-heap
                    queue.push(pair[double, npy_intp](-nd, u))
                elif nd <= distance[u] + tolerance:
                    total[u] += value
                    count[u] += 1

    values = np.full(n, np.nan)
    reached = count_arr > 0
    values[reached] = total_arr[reached] / count_arr[reached]
    return distance_arr.reshape(P, R, C), values.reshape(P, R, C)

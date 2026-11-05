# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, language_level=3
"""Lee et al. (1994) 3D thinning of labelled volumes, run a bounded number of
iterations at a time so it can be done blockwise.

Adapted from scikit-image's ``skimage/morphology/_skeletonize_lee_cy.pyx.in``
(v0.25.2), Ignacio Arganda-Carreras's BSD release of Skeletonize3D; its
licence is reproduced below. The changes follow COSEM's blockwise thinning
(cosem-segmentation-analysis's ``SparkTopologicalThinning``):

- The sequential re-check of each border direction runs over the 8 parity
  subfields of the *global* voxel index, one after another. Two voxels of one
  subfield are never 26-neighbours, so the order within a subfield does not
  matter, and one iteration (6 directions of 8 subfields) only reaches 48
  voxels. A block read with a ``48 * iterations + 2`` voxel halo therefore
  thins its interior exactly as thinning the whole volume would.
- The re-check tests Euler invariance as well as simplicity (scikit-image's
  only tests simplicity), so a candidate whose neighbours were deleted earlier
  in the same pass cannot change the topology.
- A medial-surface mode keeps surface end points (Lee's Definition 1, with
  COSEM's planes: the 3 face and 3 diagonal planes of an octant through the
  voxel) instead of curve end points.
- A voxel's neighbours are the voxels of its own label, so objects that touch
  are thinned as if each were alone.
- Neighbourhood numbering and border order are COSEM's (x the last axis), so
  results match it voxel for voxel.
- Only voxels with a 6-neighbour of another label are scanned (the "front"),
  which gives the same candidates as scanning everything.

As in scikit-image, the outermost layer of the array is never thinned: the
caller pads it.

scikit-image licence (BSD-3-Clause), for the code adapted from it:

Copyright: 2009-2022 the scikit-image team

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of the University nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

References
----------
.. [Lee94] T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models
       via 3-D medial surface/axis thinning algorithms.
       Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.
"""

from libcpp.vector cimport vector

import numpy as np
from numpy cimport npy_intp, npy_uint8, npy_uint16, npy_uint32, npy_uint64

ctypedef fused label_t:
    npy_uint8
    npy_uint16
    npy_uint32
    npy_uint64


# How far one iteration can reach: 6 directions of 8 subfields.
REACH_PER_ITERATION = 48


def halo_for(iterations):
    """The halo a block needs to thin its interior exactly for ``iterations``
    iterations: their reach plus the layer a candidate's neighbourhood reads
    and the outermost layer, which is never thinned."""
    return REACH_PER_ITERATION * int(iterations) + 2


# Lee's Euler characteristic change (column delta G_26 of Table 2), by octant
# configuration. Only odd indices (the centre set) occur.
cdef int EULER_LUT[256]
_EULER_ODD = [
    1, -1, -1, 1, -3, -1, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1, -3, -1,
    3, 1, 1, -1, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1, -3, 3, -1, 1, 1,
    3, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1, 1, 3, 3, 1, 5, 3, 3, 1,
    -1, 1, 1, -1, 3, 1, 1, -1, -7, -1, -1, 1, -3, -1, -1, 1, -1,
    1, 1, -1, 3, 1, 1, -1, -3, -1, 3, 1, 1, -1, 3, 1, -1, 1, 1,
    -1, 3, 1, 1, -1, -3, 3, -1, 1, 1, 3, -1, 1, -1, 1, 1, -1, 3,
    1, 1, -1, 1, 3, 3, 1, 5, 3, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1,
]
for _i in range(256):
    EULER_LUT[_i] = 0
for _i, _v in enumerate(_EULER_ODD):
    EULER_LUT[2 * _i + 1] = _v

cdef int POPCOUNT[256]
for _i in range(256):
    POPCOUNT[_i] = bin(_i).count("1")

# The 7 neighbourhood indices of each octant (NEB, NWB, SEB, SWB, NEU, NWU,
# SEU, SWU), from the most to the least significant bit of its configuration;
# bit 0 is the centre.
cdef int OCTANT_IDX[8][7]
_OCTANT_IDX = [
    [2, 1, 11, 10, 5, 4, 14],
    [0, 9, 3, 12, 1, 10, 4],
    [8, 7, 17, 16, 5, 4, 14],
    [6, 15, 7, 16, 3, 12, 4],
    [20, 23, 19, 22, 11, 14, 10],
    [18, 21, 9, 12, 19, 22, 10],
    [26, 23, 17, 14, 25, 22, 16],
    [24, 25, 15, 16, 21, 22, 12],
]
for _o in range(8):
    for _j in range(7):
        OCTANT_IDX[_o][_j] = _OCTANT_IDX[_o][_j]

# Octree of [Lee94] Figs. 6 and 7 for counting the components left in the
# neighbourhood without its centre (26 points, the centre dropped): each
# octant's points, and for each point the other octants that contain it.
cdef int OCTREE_IDX[8][7]
cdef int OCTREE_NEXT[8][7][3]
_OCTREE = [
    ([0, 1, 3, 4, 9, 10, 12], [[], [2], [3], [2, 3, 4], [5], [2, 5, 6], [3, 5, 7]]),
    ([1, 4, 10, 2, 5, 11, 13], [[1], [1, 3, 4], [1, 5, 6], [], [4], [6], [4, 6, 8]]),
    ([3, 4, 12, 6, 7, 14, 15], [[1], [1, 2, 4], [1, 5, 7], [], [4], [7], [4, 7, 8]]),
    ([4, 5, 13, 7, 15, 8, 16], [[1, 2, 3], [2], [2, 6, 8], [3], [3, 7, 8], [], [8]]),
    ([9, 10, 12, 17, 18, 20, 21], [[1], [1, 2, 6], [1, 3, 7], [], [6], [7], [6, 7, 8]]),
    ([10, 11, 13, 18, 21, 19, 22], [[1, 2, 5], [2], [2, 4, 8], [5], [5, 7, 8], [], [8]]),
    ([12, 14, 15, 20, 21, 23, 24], [[1, 3, 5], [3], [3, 4, 8], [5], [5, 6, 8], [], [8]]),
    ([13, 15, 16, 21, 22, 24, 25], [[2, 4, 6], [3, 4, 7], [4], [5, 6, 7], [6], [7], []]),
]
for _o, (_points, _nexts) in enumerate(_OCTREE):
    for _j in range(7):
        OCTREE_IDX[_o][_j] = _points[_j]
        for _k in range(3):
            OCTREE_NEXT[_o][_j][_k] = _nexts[_j][_k] - 1 if _k < len(_nexts[_j]) else -1

# The octant to start labelling a point from.
cdef int START_OCTANT[26]
for _i in range(26):
    START_OCTANT[_i] = next(_o for _o, (_points, _) in enumerate(_OCTREE) if _i in _points)


cdef inline int octant_configuration(const npy_uint8* nb, int octant) noexcept nogil:
    cdef int j, n = 1
    for j in range(7):
        if nb[OCTANT_IDX[octant][j]]:
            n |= 1 << (7 - j)
    return n


cdef inline bint is_euler_invariant(const npy_uint8* nb) noexcept nogil:
    """Deleting the centre keeps the Euler characteristic (condition 1)."""
    cdef int o, s = 0
    for o in range(8):
        s += EULER_LUT[octant_configuration(nb, o)]
    return s == 0


cdef inline bint is_endpoint(const npy_uint8* nb) noexcept nogil:
    """Exactly one 26-neighbour (the centre is counted too)."""
    cdef int i, s = 0
    for i in range(27):
        s += nb[i]
    return s == 2


cdef inline int count_neighbours(const npy_uint8* nb) noexcept nogil:
    cdef int i, s = 0
    for i in range(27):
        s += nb[i]
    return s - nb[13]


cdef inline bint is_surface_endpoint(const npy_uint8* nb) noexcept nogil:
    """Every octant is planar: one of the 3 face or 3 diagonal planes through
    the centre, or the centre and at most 2 more points."""
    cdef int o, n
    for o in range(8):
        n = octant_configuration(nb, o)
        if n == 51 or n == 15 or n == 85 or n == 195 or n == 165 or n == 153:
            continue
        if POPCOUNT[n] - 1 >= 3:
            return False
    return True


cdef bint is_simple_point(const npy_uint8* nb) noexcept nogil:
    """Deleting the centre leaves its neighbourhood one component (or none):
    conditions 2 and 3, "N(v)_labeling" of [Lee94]."""
    cdef npy_uint8 cube[26]
    cdef int stack[128]
    cdef int i, j, k, top, octant, idx, nxt
    cdef npy_uint8 label = 2
    for i in range(13):
        cube[i] = nb[i]
    for i in range(13, 26):
        cube[i] = nb[i + 1]
    for i in range(26):
        if cube[i] != 1:
            continue
        if label == 3:
            # a second component
            return False
        top = 0
        stack[top] = START_OCTANT[i]
        top += 1
        while top > 0:
            top -= 1
            octant = stack[top]
            for j in range(7):
                idx = OCTREE_IDX[octant][j]
                if cube[idx] != 1:
                    continue
                cube[idx] = label
                for k in range(3):
                    nxt = OCTREE_NEXT[octant][j][k]
                    if nxt >= 0:
                        stack[top] = nxt
                        top += 1
        label += 1
    return True


cdef inline void get_neighbourhood(
    const label_t* lab, npy_intp v, const npy_intp* off, label_t L, npy_uint8* nb
) noexcept nogil:
    cdef int i
    for i in range(27):
        nb[i] = lab[v + off[i]] == L


cdef inline void neighbour_offsets(npy_intp sp, npy_intp sr, npy_intp* off) noexcept nogil:
    # COSEM's numbering: index (dx + 1) + 3 (dy + 1) + 9 (dz + 1), x the last axis
    cdef int dz, dy, dx
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                off[(dx + 1) + 3 * (dy + 1) + 9 * (dz + 1)] = dz * sp + dy * sr + dx


def thin(label_t[:, :, ::1] labels not None, offset=(0, 0, 0), bint medial_surface=False,
         int max_iterations=0):
    """Thin ``labels`` in place, each label as if it were alone, until an
    iteration deletes nothing or ``max_iterations`` iterations have run (no
    limit when it is 0). ``offset`` is the global index of ``labels[0, 0, 0]``,
    which fixes the re-check's subfield order. The outermost layer is never
    thinned.

    Returns ``(iterations, deleted, converged)``: the iterations that deleted
    something, the voxels deleted, and whether an iteration deleted nothing.
    """
    cdef:
        npy_intp P = labels.shape[0], R = labels.shape[1], C = labels.shape[2]
        npy_intp sp = R * C, sr = C
        npy_intp op = offset[0], orr = offset[1], oc = offset[2]
        npy_intp off[27]
        npy_intp border_off[6]
        npy_uint8 nb[27]
        label_t* lab
        npy_uint8* in_front
        vector[npy_intp] front, kept
        vector[npy_intp] candidates[8]
        npy_intp v, u, p, r, c, rem, k
        npy_intp deleted = 0, deleted_now
        int iterations = 0, b, s, a
        bint converged = False
        label_t L

    if P < 3 or R < 3 or C < 3:
        return 0, 0, True
    lab = &labels[0, 0, 0]
    flags = np.zeros(P * R * C, dtype=np.uint8)
    cdef npy_uint8[::1] flags_view = flags
    in_front = &flags_view[0]
    neighbour_offsets(sp, sr, off)
    # COSEM's border order: up (+z), bottom (-z), north (-y), south (+y), west (-x), east (+x)
    border_off[0] = sp
    border_off[1] = -sp
    border_off[2] = -sr
    border_off[3] = sr
    border_off[4] = -1
    border_off[5] = 1

    with nogil:
        for p in range(1, P - 1):
            for r in range(1, R - 1):
                for c in range(1, C - 1):
                    v = p * sp + r * sr + c
                    L = lab[v]
                    if L == 0:
                        continue
                    if (lab[v - 1] != L or lab[v + 1] != L or lab[v - sr] != L
                            or lab[v + sr] != L or lab[v - sp] != L or lab[v + sp] != L):
                        front.push_back(v)
                        in_front[v] = 1

        while max_iterations <= 0 or iterations < max_iterations:
            deleted_now = 0
            for b in range(6):
                # candidates: border points of this direction that are deletable
                kept.clear()
                for s in range(8):
                    candidates[s].clear()
                for k in range(<npy_intp>front.size()):
                    v = front[k]
                    L = lab[v]
                    if L == 0:
                        in_front[v] = 0
                        continue
                    kept.push_back(v)
                    if lab[v + border_off[b]] == L:
                        continue
                    get_neighbourhood(lab, v, off, L, nb)
                    if medial_surface:
                        if is_surface_endpoint(nb):
                            continue
                    elif is_endpoint(nb):
                        continue
                    if not is_euler_invariant(nb) or not is_simple_point(nb):
                        continue
                    p = v // sp
                    rem = v - p * sp
                    r = rem // sr
                    c = rem - r * sr
                    s = ((c + oc) & 1) + 2 * ((r + orr) & 1) + 4 * ((p + op) & 1)
                    candidates[s].push_back(v)
                front.swap(kept)

                # sequential re-check, one subfield at a time
                for s in range(8):
                    for k in range(<npy_intp>candidates[s].size()):
                        v = candidates[s][k]
                        L = lab[v]
                        get_neighbourhood(lab, v, off, L, nb)
                        if not (is_simple_point(nb) and is_euler_invariant(nb)):
                            continue
                        lab[v] = 0
                        deleted_now += 1
                        p = v // sp
                        rem = v - p * sp
                        r = rem // sr
                        c = rem - r * sr
                        for a in range(6):
                            if a == 0:
                                if p + 1 > P - 2:
                                    continue
                                u = v + sp
                            elif a == 1:
                                if p - 1 < 1:
                                    continue
                                u = v - sp
                            elif a == 2:
                                if r + 1 > R - 2:
                                    continue
                                u = v + sr
                            elif a == 3:
                                if r - 1 < 1:
                                    continue
                                u = v - sr
                            elif a == 4:
                                if c + 1 > C - 2:
                                    continue
                                u = v + 1
                            else:
                                if c - 1 < 1:
                                    continue
                                u = v - 1
                            if lab[u] != 0 and not in_front[u]:
                                front.push_back(u)
                                in_front[u] = 1
            if deleted_now == 0:
                converged = True
                break
            iterations += 1
            deleted += deleted_now

    return iterations, deleted, converged


def faces_are_final(label_t[:, :, ::1] labels not None, lo, hi, bint medial_surface=False):
    """Whether every foreground voxel on the faces of the box ``[lo, hi)`` is a
    border point that no later iteration can delete: a curve end point, a
    curve voxel with 2 neighbours that is neither simple nor Euler invariant,
    or (``medial_surface``) a surface end point. Deletions only keep such a
    voxel final, so the box can then be thinned with no halo. The box must sit
    at least one voxel inside ``labels``."""
    cdef:
        npy_intp P = labels.shape[0], R = labels.shape[1], C = labels.shape[2]
        npy_intp sp = R * C, sr = C
        npy_intp p0 = lo[0], r0 = lo[1], c0 = lo[2]
        npy_intp p1 = hi[0], r1 = hi[1], c1 = hi[2]
        npy_intp off[27]
        npy_uint8 nb[27]
        npy_intp p, r, c, v, step
        label_t* lab
        label_t L
        int n
        bint final = True
    if p0 < 1 or r0 < 1 or c0 < 1 or p1 > P - 1 or r1 > R - 1 or c1 > C - 1:
        raise ValueError("the box must sit at least one voxel inside the array")
    if p1 <= p0 or r1 <= r0 or c1 <= c0:
        return True
    lab = &labels[0, 0, 0]
    neighbour_offsets(sp, sr, off)
    with nogil:
        for p in range(p0, p1):
            if not final:
                break
            for r in range(r0, r1):
                if not final:
                    break
                # inside the box only its first and last columns are on a face
                if p == p0 or p == p1 - 1 or r == r0 or r == r1 - 1:
                    step = 1
                else:
                    step = c1 - 1 - c0
                    if step < 1:
                        step = 1
                c = c0
                while c < c1:
                    v = p * sp + r * sr + c
                    c += step
                    L = lab[v]
                    if L == 0:
                        continue
                    if not (lab[v - 1] != L or lab[v + 1] != L or lab[v - sr] != L
                            or lab[v + sr] != L or lab[v - sp] != L or lab[v + sp] != L):
                        final = False
                        break
                    get_neighbourhood(lab, v, off, L, nb)
                    if medial_surface:
                        if not is_surface_endpoint(nb):
                            final = False
                            break
                    else:
                        n = count_neighbours(nb)
                        if not (n == 1 or (n == 2 and not is_simple_point(nb)
                                           and not is_euler_invariant(nb))):
                            final = False
                            break
    return final


# For tests: the neighbourhood predicates on a 27-voxel neighbourhood in
# COSEM's numbering (index 13 the centre).
def _is_simple_point(neighbourhood):
    cdef npy_uint8[::1] nb = np.ascontiguousarray(neighbourhood, dtype=np.uint8).ravel()
    return bool(is_simple_point(&nb[0]))


def _is_euler_invariant(neighbourhood):
    cdef npy_uint8[::1] nb = np.ascontiguousarray(neighbourhood, dtype=np.uint8).ravel()
    return bool(is_euler_invariant(&nb[0]))


def _is_surface_endpoint(neighbourhood):
    cdef npy_uint8[::1] nb = np.ascontiguousarray(neighbourhood, dtype=np.uint8).ravel()
    return bool(is_surface_endpoint(&nb[0]))

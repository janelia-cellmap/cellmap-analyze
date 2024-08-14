# bresenham3d.pyx

from cpython cimport bool
from libc.stdlib cimport abs
from typing import List, Tuple

cdef inline bint append_if_not_masked(int x, int y, int z, list points, mask=None):
    if mask is not None and mask[x, y, z]:
        return False
    else:
        points.append((x, y, z))
        return True

def bresenham3DWithMask(
    int x1, int y1, int z1, int x2, int y2, int z2, mask=None
) -> List[Tuple[int, int, int]]:

    cdef list listOfPoints = []
    if not append_if_not_masked(x1, y1, z1, listOfPoints, mask=mask):
        return []
    cdef int dx = abs(x2 - x1)
    cdef int dy = abs(y2 - y1)
    cdef int dz = abs(z2 - z1)
    cdef int xs = 1 if x2 > x1 else -1
    cdef int ys = 1 if y2 > y1 else -1
    cdef int zs = 1 if z2 > z1 else -1

    cdef int p1, p2

    # Driving axis is X-axis
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz

            if not append_if_not_masked(x1, y1, z1, listOfPoints, mask=mask):
                return []

    # Driving axis is Y-axis
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz

            if not append_if_not_masked(x1, y1, z1, listOfPoints, mask=mask):
                return []

    # Driving axis is Z-axis
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx

            if not append_if_not_masked(x1, y1, z1, listOfPoints, mask=mask):
                return []
    return listOfPoints
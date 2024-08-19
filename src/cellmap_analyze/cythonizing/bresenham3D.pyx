# bresenham3d.pyx

from cpython cimport bool
from libc.stdlib cimport abs
from typing import List, Tuple

cdef inline bint append_if_not_masked(int x, int y, int z, list points, unsigned char[:,:,:] mask=None):
    if mask is not None and mask[x,y,z]:
        return False
    else:
        points.append((x, y, z))
        return True

def bresenham3DWithMaskSingle(
    int x1, int y1, int z1, int x2, int y2, int z2, unsigned char[:,:,:] mask=None
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

def bresenham3DWithMask(
    int[:, :] starts_array, int[:, :] ends_array, unsigned char[:,:,:] mask = None
) -> List[Tuple[int, int, int]]:
    cdef list output_list = []
    for current_start, current_end in zip(starts_array, ends_array):
        output_list += bresenham3DWithMaskSingle(
            current_start[0], current_start[1], current_start[2],
            current_end[0], current_end[1], current_end[2],
            mask=mask
        )
    return output_list

cdef fused uint_t:
    unsigned char
    unsigned short
    unsigned int
    unsigned long long

def find_boundary(
    uint_t[:, :, :] volume, unsigned char[:,:,:] surface_voxels
):
   
    cdef int x, y, z
    cdef int nx = volume.shape[0]
    cdef int ny = volume.shape[1]
    cdef int nz = volume.shape[2]
    cdef uint_t voxel_value
    # Iterate over each voxel in the volume
    for x in range(1, nx-1):
        for y in range(1, ny-1):
            for z in range(1, nz-1):
                # Get the value of the current voxel
                voxel_value = volume[x, y, z]
                
                # Check the 6 neighbors
                if voxel_value>0:
                    if (volume[x-1, y, z] != voxel_value or
                        volume[x+1, y, z] != voxel_value or
                        volume[x, y-1, z] != voxel_value or
                        volume[x, y+1, z] != voxel_value or
                        volume[x, y, z-1] != voxel_value or
                        volume[x, y, z+1] != voxel_value):
                        # Add the surface voxel coordinates to the list
                        surface_voxels[x, y, z] = 1
        
# bresenham3d.pyx

from cpython cimport bool
from libc.stdlib cimport abs
from typing import List, Tuple

cdef inline bint append_if_not_masked(int x, int y, int z, int idx, list points, unsigned char[:,:,:] mask=None):
    if mask is not None and mask[x,y,z]:
        return False
    else:
        points[idx] = (x,y,z)
        return True

def bresenham3DWithMaskSingle(
    int x1, int y1, int z1, int x2, int y2, int z2,
    list points,
    unsigned char[:,:,:] mask=None
):

    cdef int idx = 0
    if not append_if_not_masked(x1, y1, z1, idx, points, mask=mask):
        return -1
    idx+=1

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
            
            if not append_if_not_masked(x1, y1, z1, idx, points, mask=mask):
                return -1
            idx+=1

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
            
            if not append_if_not_masked(x1, y1, z1, idx, points, mask=mask):
                return -1
            idx+=1

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

            if not append_if_not_masked(x1, y1, z1, idx, points, mask=mask):
                return -1
            idx+=1
    
    return idx

cdef inline int bresenham3DWithMaskSingleInline(
    int x1, int y1, int z1, int x2, int y2, int z2,
    list points,
    unsigned char[:,:,:] mask=None
):

    cdef int idx = 0
    if not append_if_not_masked(x1, y1, z1, idx, points, mask=mask):
        return -1
    idx+=1

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
            
            if not append_if_not_masked(x1, y1, z1, idx, points, mask=mask):
                return -1
            idx+=1

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
            
            if not append_if_not_masked(x1, y1, z1, idx, points, mask=mask):
                return -1
            idx+=1

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

            if not append_if_not_masked(x1, y1, z1, idx, points, mask=mask):
                return -1
            idx+=1
    
    return idx

def bresenham3DWithMask(
    int[:, :] starts_array, int[:, :] ends_array, unsigned char[:,:,:] mask = None
):
    cdef list output_list = []
    cdef int rows = starts_array.shape[0]
    cdef set output_set = set()
    cdef list points = []

    for i in range(rows):
    
        idx = bresenham3DWithMaskSingle(
        starts_array[i][0], starts_array[i][1], starts_array[i][2],
        ends_array[i][0], ends_array[i][1], ends_array[i][2],
        points,
        mask=mask)
        output_list+=points
      
    return output_list

def bresenham_3D_lines(list contact_voxels_list_of_lists,
    long  [:,:] object_1_surface_voxel_coordinates,
    long  [:,:] object_2_surface_voxel_coordinates,
    unsigned char [:,:,:] current_pair_contact_sites,
    int max_num_voxels,
    unsigned char[:,:,:] mask):
    cdef set all_valid_voxels = set()
    points=[() for _ in range(max_num_voxels)]
    cdef int i,j,x,y,z
    cdef list sublist
    cdef tuple point
    cdef bool found_contact_voxels = False
    for i, sublist in enumerate(contact_voxels_list_of_lists):
        for j in sublist:
            contact_voxel_1 = object_1_surface_voxel_coordinates[i]
            contact_voxel_2 = object_2_surface_voxel_coordinates[j]
            if (
                current_pair_contact_sites[
                    contact_voxel_1[0], contact_voxel_1[1], contact_voxel_1[2]
                ]
                or current_pair_contact_sites[
                    contact_voxel_2[0], contact_voxel_2[1], contact_voxel_2[2]
                ]
            ):
                continue
            idx = bresenham3DWithMaskSingleInline(
                contact_voxel_1[0],
                contact_voxel_1[1],
                contact_voxel_1[2],
                contact_voxel_2[0],
                contact_voxel_2[1],
                contact_voxel_2[2],
                points,
                mask=mask,
            )
            if idx>0:
                found_contact_voxels = True
                all_valid_voxels.update(points[:idx])
    for (x,y,z) in all_valid_voxels:
        current_pair_contact_sites[x,y,z] = 1
    return found_contact_voxels

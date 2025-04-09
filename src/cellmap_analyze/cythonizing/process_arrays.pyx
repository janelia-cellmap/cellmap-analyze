# bresenham3d.pyx

from cpython cimport bool
from libc.stdlib cimport abs
import numpy as np

cdef fused uint_t:
    unsigned char
    unsigned short
    unsigned int
    unsigned long
    unsigned long long

cdef fused uint_t1:
    unsigned char
    unsigned short
    unsigned int
    unsigned long
    unsigned long long

def initialize_contact_site_array(
    uint_t[:, :, :] organelle_1, 
    uint_t1[:, :, :] organelle_2, 
    unsigned char[:, :, :] surface_voxels_1, 
    unsigned char[:, :, :] surface_voxels_2, 
    unsigned char[:, :, :] mask,
    unsigned char[:, :, :] initialized_contact_sites,
    bint mask_out_surface_voxels):

    cdef int x, y, z
    cdef int nx = initialized_contact_sites.shape[0]
    cdef int ny = initialized_contact_sites.shape[1]
    cdef int nz = initialized_contact_sites.shape[2]
    cdef uint_t organelle_1_value
    cdef uint_t1 organelle_2_value
    cdef unsigned char surface_voxel_1_value, surface_voxel_2_value

    # Iterate over each voxel in the volume
    for x in range(1, nx-1):
        for y in range(1, ny-1):
            for z in range(1, nz-1):
                # Get the value of the current voxel
                organelle_1_value = organelle_1[x, y, z]
                organelle_2_value = organelle_2[x, y, z]
                if organelle_1_value and organelle_2_value:
                    initialized_contact_sites[x, y, z] = 1
                    
                if organelle_1_value>0:
                    if (organelle_1[x-1, y, z] != organelle_1_value or
                        organelle_1[x+1, y, z] != organelle_1_value or
                        organelle_1[x, y-1, z] != organelle_1_value or
                        organelle_1[x, y+1, z] != organelle_1_value or
                        organelle_1[x, y, z-1] != organelle_1_value or
                        organelle_1[x, y, z+1] != organelle_1_value):
                        # Add the surface voxel coordinates to the list
                        surface_voxels_1[x, y, z] = 1


                if organelle_2_value>0:
                    if (organelle_2[x-1, y, z] != organelle_2_value or
                        organelle_2[x+1, y, z] != organelle_2_value or
                        organelle_2[x, y-1, z] != organelle_2_value or
                        organelle_2[x, y+1, z] != organelle_2_value or
                        organelle_2[x, y, z-1] != organelle_2_value or
                        organelle_2[x, y, z+1] != organelle_2_value):
                        # Add the surface voxel coordinates to the list
                        surface_voxels_2[x, y, z] = 1

                surface_voxel_1_value = surface_voxels_1[x, y, z]
                surface_voxel_2_value = surface_voxels_2[x, y, z]          

                if mask_out_surface_voxels:
                    mask[x, y, z] = organelle_1_value>0 or organelle_2_value>0
                else:
                    mask[x, y, z] = (organelle_1_value>0 and surface_voxel_1_value==0) |  (organelle_2_value>0 and surface_voxel_2_value==0)
import numpy as np
cimport numpy as np
from libc.math cimport pow

# Assuming the ContactingOrganelleInformation class is defined elsewhere
cdef class ContactingOrganelleInformation:
    # Implement the class definition and methods as needed
    pass

cdef double[:,:,:,::1] trim_array(double[:,:,:,::1] arr, int trim)

cpdef get_contacting_organelle_information(
    np.ndarray[np.int8_t, ndim=3] contact_sites,
    np.ndarray[np.int8_t, ndim=3] contacting_organelle,
    double voxel_edge_length=1.0, int trim=1
):
    cdef double voxel_face_area = pow(voxel_edge_length, 2)
    
    # surface_areas should be a 3D numpy array, replace with your actual implementation
    cdef np.ndarray[np.float64_t, ndim=3] surface_areas = calculate_surface_areas_voxelwise(contacting_organelle, voxel_face_area)

    # Trim arrays to consider only the current block
    surface_areas = trim_array(surface_areas, trim)
    contact_sites = trim_array(contact_sites, trim)
    contacting_organelle = trim_array(contacting_organelle, trim)

    # Limit to where contact sites overlap with objects
    cdef np.ndarray[np.bool_t, ndim=3] mask = np.logical_and(contact_sites > 0, contacting_organelle > 0)
    contact_sites = contact_sites[mask].ravel()
    contacting_organelle = contacting_organelle[mask].ravel()
    surface_areas = surface_areas[mask].ravel()

    # Use np.unique to get unique groups and counts
    cdef np.ndarray[np.int64_t, ndim=2] groups
    cdef np.ndarray[np.int64_t, ndim=1] counts
    groups, counts = np.unique(
        np.array([contact_sites, contacting_organelle, surface_areas], dtype=np.float64),
        axis=1,
        return_counts=True,
    )
    
    cdef np.ndarray[np.int64_t, ndim=1] contact_site_ids = groups[0]
    cdef np.ndarray[np.int64_t, ndim=1] contacting_ids = groups[1]
    cdef np.ndarray[np.float64_t, ndim=1] surface_areas_result = groups[2] * counts

    contact_site_to_contacting_information_dict = {}

    # Iterating over the results
    cdef int i
    cdef int n = len(contact_site_ids)
    for i in range(n):
        contact_site_id = contact_site_ids[i]
        contacting_id = contacting_ids[i]
        surface_area = surface_areas_result[i]
        
        coi = contact_site_to_contacting_information_dict.get(
            contact_site_id,
            ContactingOrganelleInformation(),
        )
        coi += ContactingOrganelleInformation({contacting_id: surface_area})
        contact_site_to_contacting_information_dict[contact_site_id] = coi

    return contact_site_to_contacting_information_dict
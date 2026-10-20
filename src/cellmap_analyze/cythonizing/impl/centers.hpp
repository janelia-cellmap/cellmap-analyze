// The following code was taken from funlib.evaluate: https://github.com/funkelab/funlib.evaluate/blob/master/funlib/evaluate/impl/centers.hpp

#ifndef IMPL_CENTERS_H__
#define IMPL_CENTERS_H__

#include <array>
#include <map>

struct Center {
    Center() {
        z = 0.0;
        y = 0.0;
        x = 0.0;
        n = 0;
        sum_r2 = 0.0;
        szz = syy = sxx = szy = szx = syx = 0.0;
    }

    double z, y, x;
    size_t n;
    double sum_r2;  // sum of squared distances from origin = szz + syy + sxx
    // Second-moment tensor about the origin: the six independent sums of
    // products. Together with n and the centre of mass these give the
    // covariance, whose eigenvalues size an object's principal axes and whose
    // eigenvectors orient them. Sums, so they add across blocks as sum_r2 does.
    double szz, syy, sxx, szy, szx, syx;
};

/**
 * Compute per-label centers and optionally sum of squared distances (r^2).
 * @param size_z  number of slices in z-dimension
 * @param size_y  number of rows in y-dimension
 * @param size_x  number of columns in x-dimension
 * @param labels  pointer to a flat array of length size_z*size_y*size_x
 * @param compute_sum_r2  if true, accumulate sum of (z^2 + y^2 + x^2) per label
 * @param compute_second_moments  if true, accumulate the six sums of products
 *        (zz, yy, xx, zy, zx, yx) per label
 * @param voxel_size  pointer to array of length 3 with voxel sizes [z, y, x]
 * @return map from label to its Center (with COM and n, and sum_r2)
 */

template <typename T>
std::map<T, Center>
centers(
        size_t size_z,
        size_t size_y,
        size_t size_x,
        const T* labels,
        bool compute_sum_r2 = false,
        bool compute_second_moments = false,
        bool center_on_voxels = true,
        const double* voxel_size = nullptr,  // pointer to [z, y, x] voxel sizes
        const double* offset = nullptr   // pointer, default nullptr => {0,0,0}
    ) {
    std::map<T, Center> centers;
    size_t total = size_z * size_y * size_x;
    std::array<int, 3> pos = {{0, 0, 0}};
    double extra_addon = center_on_voxels ? 0.5 : 0.0;

    // Default voxel_size to (1, 1, 1) if not provided
    double default_voxel_size[3] = {1.0, 1.0, 1.0};
    if (voxel_size == nullptr) {
        voxel_size = default_voxel_size;
    }

    // Default offset to (0, 0, 0) if not provided
    double default_offset[3] = {0.0, 0.0, 0.0};
    if (offset == nullptr) {
        offset = default_offset;
    }

    for (size_t i = 0; i < total; ++i) {
        T l = labels[i];
        if (l > 0) {
            auto& c = centers[l];
            // Apply per-axis voxel size
            double p0 = (pos[0] + extra_addon) * voxel_size[0] + offset[0];
            double p1 = (pos[1] + extra_addon) * voxel_size[1] + offset[1];
            double p2 = (pos[2] + extra_addon) * voxel_size[2] + offset[2];

            c.z += p0;
            c.y += p1;
            c.x += p2;
            c.n++;
            if (compute_sum_r2) {
                c.sum_r2 += p0 * p0 + p1 * p1 + p2 * p2;
            }
            if (compute_second_moments) {
                c.szz += p0 * p0;
                c.syy += p1 * p1;
                c.sxx += p2 * p2;
                c.szy += p0 * p1;
                c.szx += p0 * p2;
                c.syx += p1 * p2;
            }
        }

        // increment position indices
        if (++pos[2] >= static_cast<int>(size_x)) {
            pos[2] = 0;
            if (++pos[1] >= static_cast<int>(size_y)) {
                pos[1] = 0;
                ++pos[0];
            }
        }
    }

    // finalize center-of-mass
    for (auto& p : centers) {
        Center& c = p.second;
        if (c.n > 0) {
            c.z /= c.n;
            c.y /= c.n;
            c.x /= c.n;
        }
    }

    return centers;
}
#endif // IMPL_CENTERS_H__

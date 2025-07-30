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
    }

    double z, y, x;
    size_t n;
    double sum_r2;  // sum of squared distances from origin
};

/**
 * Compute per-label centers and optionally sum of squared distances (r^2).
 * @param size_z  number of slices in z-dimension
 * @param size_y  number of rows in y-dimension
 * @param size_x  number of columns in x-dimension
 * @param labels  pointer to a flat array of length size_z*size_y*size_x
 * @param compute_sum_r2  if true, accumulate sum of (z^2 + y^2 + x^2) per label
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
        bool center_on_voxels = true,
        double voxel_edge_length = 1.0,
        const double* offset = nullptr   // pointer, default nullptr => {0,0,0}
    ) {
    std::map<T, Center> centers;
    size_t total = size_z * size_y * size_x;
    std::array<int, 3> pos = {{0, 0, 0}};
    double extra_addon = center_on_voxels ? 0.5 : 0.0;
    for (size_t i = 0; i < total; ++i) {
        T l = labels[i];
        if (l > 0) {
            auto& c = centers[l];
            double p0 = (pos[0] + extra_addon) * voxel_edge_length + offset[0];
            double p1 = (pos[1] + extra_addon) * voxel_edge_length + offset[1];
            double p2 = (pos[2] + extra_addon) * voxel_edge_length + offset[2];

            c.z += p0;
            c.y += p1;
            c.x += p2;
            c.n++;
            if (compute_sum_r2) {
                c.sum_r2 +=
                    static_cast<double>(p0) * p0 +
                    static_cast<double>(p1) * p1 +
                    static_cast<double>(p2) * p2;
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

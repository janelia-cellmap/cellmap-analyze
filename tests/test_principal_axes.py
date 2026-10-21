"""Principal axes from the blockwise second-moment tensor."""
import numpy as np
import pytest

from cellmap_analyze.cythonizing.centers import find_centers
from cellmap_analyze.util.information_holders import ObjectInformation


def _object(labels, voxel_size=(1.0, 1.0, 1.0), offset=(0.0, 0.0, 0.0)):
    """ObjectInformation for label 1 of `labels`, as measure builds one."""
    ids = np.array([1])
    coords, sum_r2, sum_m2 = find_centers(
        labels, ids, compute_sum_r2=True, compute_second_moments=True,
        voxel_size=np.asarray(voxel_size, float), offset=np.asarray(offset, float),
    )
    n = int((labels == 1).sum())
    return ObjectInformation(counts=n, volume=float(n), com=coords[0],
                             sum_r2=sum_r2[0], sum_m2=sum_m2[0])


def test_axes_of_a_box():
    # A box N voxels long has variance (N^2 - 1)/12 along that side, and the
    # lengths are sqrt(20 * variance).
    labels = np.zeros((30, 10, 10), np.uint64)
    labels[5:25, 3:7, 4:6] = 1          # 20 x 4 x 2 in z, y, x
    oi = _object(labels)
    lengths, vectors = oi.principal_axes
    variances = (np.array([20, 4, 2]) ** 2 - 1) / 12
    assert np.allclose(lengths, np.sqrt(20 * variances))
    assert np.allclose(np.abs(vectors[:, 0]), [1, 0, 0])      # longest axis is z
    assert np.isclose(oi.elongation, lengths[0] / lengths[1])
    assert np.isclose(oi.flatness, lengths[1] / lengths[2])


def test_tensor_adds_across_blocks():
    # The property the blockwise pass relies on: an object cut in two and
    # merged must measure the same as the whole object.
    labels = np.zeros((30, 10, 10), np.uint64)
    labels[5:25, 3:7, 4:6] = 1
    whole = _object(labels)

    cut = 15
    lower, upper = labels.copy(), labels.copy()
    lower[cut:] = 0
    upper[:cut] = 0
    # Each half is measured in its own block, with the block's offset, exactly
    # as measure does -- so the sums share one origin and can be added.
    merged = _object(lower) + _object(upper[cut:], offset=(cut, 0, 0))

    assert merged.counts == whole.counts
    assert np.allclose(merged.com, whole.com)
    assert np.allclose(merged.sum_m2, whole.sum_m2)
    assert np.allclose(merged.principal_axes[0], whole.principal_axes[0])
    assert np.allclose(np.abs(merged.principal_axes[1]), np.abs(whole.principal_axes[1]))


def test_anisotropic_voxels_measure_in_nm():
    # 10 voxels of 8 nm in z, 10 of 2 nm in x, 2 of 4 nm in y: 80 x 20 x 8 nm,
    # so the axes are z, then x, then y -- the voxel size, not the voxel count,
    # decides which is longest.
    labels = np.zeros((20, 4, 20), np.uint64)
    labels[5:15, 1:3, 5:15] = 1
    oi = _object(labels, voxel_size=(8.0, 4.0, 2.0))
    lengths, vectors = oi.principal_axes
    assert np.allclose(np.abs(vectors[:, 0]), [1, 0, 0])
    variances = np.array([80**2 - 8**2, 20**2 - 2**2, 8**2 - 4**2]) / 12
    assert np.allclose(lengths, np.sqrt(20 * variances))


def test_a_sphere_measures_its_diameter():
    z, y, x = np.ogrid[:31, :31, :31]
    labels = (((z - 15) ** 2 + (y - 15) ** 2 + (x - 15) ** 2) <= 12**2).astype(np.uint64)
    oi = _object(labels)
    assert np.allclose(oi.principal_axes[0], 24, atol=0.5)
    assert oi.elongation == pytest.approx(1.0, abs=0.02)
    assert oi.flatness == pytest.approx(1.0, abs=0.02)


def test_radius_of_gyration_is_the_tensor_trace():
    labels = np.zeros((30, 10, 10), np.uint64)
    labels[5:25, 3:7, 4:6] = 1
    oi = _object(labels)
    zz, yy, xx = oi.sum_m2[:3]
    assert np.isclose(oi.sum_r2, zz + yy + xx)

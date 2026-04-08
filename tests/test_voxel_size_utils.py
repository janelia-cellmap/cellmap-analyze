"""Tests for voxel size scaling utilities."""
import numpy as np
import pytest
from cellmap_analyze.util.voxel_size_utils import (
    scale_voxel_size_to_integers,
    is_integer_voxel_size,
    compute_common_scale_factor,
)


class TestIsIntegerVoxelSize:
    def test_integers(self):
        assert is_integer_voxel_size((8, 8, 8))
        assert is_integer_voxel_size((3, 7, 5))

    def test_floats_that_are_integers(self):
        assert is_integer_voxel_size((8.0, 8.0, 8.0))

    def test_near_integers(self):
        assert is_integer_voxel_size((8.0000000001, 4.0, 4.0))

    def test_non_integers(self):
        assert not is_integer_voxel_size((3.54, 4, 4))
        assert not is_integer_voxel_size((3.5, 3.5, 3.5))


class TestScaleVoxelSizeToIntegers:
    def test_already_integer(self):
        scaled, factor = scale_voxel_size_to_integers((8, 8, 8))
        assert scaled == (8, 8, 8)
        assert factor == 1

    def test_already_integer_floats(self):
        scaled, factor = scale_voxel_size_to_integers((8.0, 8.0, 8.0))
        assert scaled == (8, 8, 8)
        assert factor == 1

    def test_simple_decimal(self):
        scaled, factor = scale_voxel_size_to_integers((3.5, 4, 4))
        assert factor == 2
        assert scaled == (7, 8, 8)

    def test_two_decimal_places(self):
        scaled, factor = scale_voxel_size_to_integers((3.54, 4, 4))
        # 3.54 = 177/50, so factor should be 50
        assert factor == 50
        assert scaled == (177, 200, 200)

    def test_all_non_integer(self):
        scaled, factor = scale_voxel_size_to_integers((3.5, 4.5, 5.5))
        assert factor == 2
        assert scaled == (7, 9, 11)

    def test_round_trip_accuracy(self):
        """Verify scaled_vs / scale_factor ≈ original_vs."""
        original = (3.54, 4, 4)
        scaled, factor = scale_voxel_size_to_integers(original)
        recovered = tuple(s / factor for s in scaled)
        for o, r in zip(original, recovered):
            assert abs(o - r) < 1e-6

    def test_repeating_decimal(self):
        """Fraction.limit_denominator should handle repeating decimals."""
        scaled, factor = scale_voxel_size_to_integers((3.333, 4, 4))
        # Should produce close-to-correct rational approximation
        recovered = tuple(s / factor for s in scaled)
        assert abs(recovered[0] - 3.333) < 0.01

    def test_mixed_anisotropic(self):
        scaled, factor = scale_voxel_size_to_integers((3.54, 8, 8))
        assert factor == 50
        assert scaled == (177, 400, 400)


class TestComputeCommonScaleFactor:
    def test_same_factors(self):
        assert compute_common_scale_factor(50, 50) == 50

    def test_one_is_one(self):
        assert compute_common_scale_factor(1, 50) == 50

    def test_both_one(self):
        assert compute_common_scale_factor(1, 1) == 1

    def test_lcm(self):
        assert compute_common_scale_factor(6, 4) == 12

    def test_three_factors(self):
        assert compute_common_scale_factor(2, 3, 5) == 30

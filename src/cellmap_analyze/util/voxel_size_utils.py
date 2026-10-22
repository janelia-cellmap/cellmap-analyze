from fractions import Fraction
from math import gcd


def _lcm(a, b):
    return a * b // gcd(a, b)


def is_integer_voxel_size(voxel_size, tolerance=1e-9):
    """Check if all components of voxel_size are effectively integers."""
    return all(abs(v - round(v)) < tolerance for v in voxel_size)


def scale_voxel_size_to_integers(voxel_size):
    """Scale a float voxel_size tuple to integers by finding a minimal common multiplier.

    Uses Fraction to convert each component to a rational number, then finds
    the LCM of all denominators to produce the smallest integer scaling.

    Args:
        voxel_size: Tuple of float voxel sizes, e.g. (3.54, 4, 4)

    Returns:
        (scaled_voxel_size, scale_factor) where:
        - scaled_voxel_size is a tuple of ints, e.g. (177, 200, 200)
        - scale_factor is the int multiplier used, e.g. 50
    """
    if is_integer_voxel_size(voxel_size):
        return tuple(int(round(v)) for v in voxel_size), 1

    fractions = [Fraction(v).limit_denominator(10000) for v in voxel_size]
    denominators = [f.denominator for f in fractions]

    scale_factor = denominators[0]
    for d in denominators[1:]:
        scale_factor = _lcm(scale_factor, d)

    scaled = tuple(int(round(float(f) * scale_factor)) for f in fractions)
    return scaled, scale_factor


def compute_common_scale_factor(*scale_factors):
    """Compute the LCM of multiple scale factors.

    When combining datasets with different scale factors, all must work
    in the same scaled coordinate space. This returns the LCM of all factors.

    Args:
        *scale_factors: Integer scale factors from different IDIs.

    Returns:
        The LCM of all scale factors.
    """
    result = scale_factors[0]
    for sf in scale_factors[1:]:
        result = _lcm(result, sf)
    return result

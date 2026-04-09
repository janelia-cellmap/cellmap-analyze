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


def read_raw_voxel_size(ds):
    """Read the original float voxel_size from zarr/n5 attributes.

    Reads directly from zarr attributes to avoid funlib.geometry.Coordinate
    truncation. Checks multiple metadata formats.

    Args:
        ds: A funlib.persistence Array (returned by open_ds)

    Returns:
        Tuple of floats representing the true voxel size.
    """
    attrs = dict(ds.data.attrs)

    # Check for our own original_voxel_size attr (written by create_multiscale_dataset)
    if "original_voxel_size" in attrs:
        return tuple(float(v) for v in attrs["original_voxel_size"])

    # Check funlib-style voxel_size attribute on the array
    if "voxel_size" in attrs:
        return tuple(float(v) for v in attrs["voxel_size"])

    # Check OME-Zarr multiscales on the array itself (rare but possible)
    if "multiscales" in attrs:
        return _extract_ome_scale(attrs)

    # Check OME-Zarr multiscales on parent group
    parent_attrs = _read_parent_attrs(ds)
    if parent_attrs and "multiscales" in parent_attrs:
        return _extract_ome_scale(parent_attrs)

    # Check N5 pixelResolution
    if "pixelResolution" in attrs:
        return tuple(float(v) for v in attrs["pixelResolution"]["dimensions"])

    # Fallback: use what funlib already parsed (may be truncated)
    return tuple(float(v) for v in ds.voxel_size)


def read_raw_offset(ds):
    """Read the original float offset from zarr/n5 attributes.

    Args:
        ds: A funlib.persistence Array (returned by open_ds)

    Returns:
        Tuple of floats representing the true offset.
    """
    attrs = dict(ds.data.attrs)

    # Check funlib-style offset attribute
    if "offset" in attrs:
        return tuple(float(v) for v in attrs["offset"])

    # Check OME-Zarr multiscales on the array
    if "multiscales" in attrs:
        translation = _extract_ome_translation(attrs)
        if translation is not None:
            return translation

    # Check parent group for OME-Zarr
    parent_attrs = _read_parent_attrs(ds)
    if parent_attrs and "multiscales" in parent_attrs:
        translation = _extract_ome_translation(parent_attrs)
        if translation is not None:
            return translation

    # Fallback
    return tuple(float(v) for v in ds.roi.offset)


def _extract_ome_scale(attrs):
    """Extract voxel size from OME-Zarr multiscales metadata."""
    transforms = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    for t in transforms:
        if t["type"] == "scale":
            return tuple(float(v) for v in t["scale"])
    raise ValueError("No scale transform found in OME-Zarr multiscales metadata")


def _extract_ome_translation(attrs):
    """Extract offset from OME-Zarr multiscales metadata."""
    transforms = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    for t in transforms:
        if t["type"] == "translation":
            return tuple(float(v) for v in t["translation"])
    return None


def _read_parent_attrs(ds):
    """Try to read attributes from the parent zarr group."""
    try:
        import zarr

        store = ds.data.store
        parent_path = "/".join(ds.data.path.split("/")[:-1]) if ds.data.path else ""
        if parent_path:
            parent = zarr.open(store, mode="r", path=parent_path)
            return dict(parent.attrs)
    except Exception:
        pass
    return None

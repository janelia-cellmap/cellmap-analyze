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


def get_multiscales(attrs):
    """Return the OME ``multiscales`` list from an attrs dict, or None.

    Handles both metadata layouts:
    - NGFF <= 0.4: ``multiscales`` sits at the top level of the attrs.
    - NGFF >= 0.5: everything is nested under an ``ome`` key, so the list
      lives at ``attrs["ome"]["multiscales"]``.
    """
    if not attrs:
        return None
    if "multiscales" in attrs:
        return attrs["multiscales"]
    ome = attrs.get("ome")
    if isinstance(ome, dict) and "multiscales" in ome:
        return ome["multiscales"]
    return None


def _read_parent_attrs(ds):
    """Try to read attributes from the parent zarr group."""
    try:
        import os
        import zarr

        # Remote opens (open_dataset for s3://, gs://, http(s)://) stash
        # the parent group's attrs as a dict directly on the
        # CellMapArray, so we don't have to go back over the wire and
        # don't have to route through zarr's fsspec backend.
        stashed = getattr(ds, "_cellmap_parent_attrs", None)
        if stashed is not None:
            return stashed

        # Preferred path: the CellMapArray was opened via ``open_dataset``,
        # which stashes the full path on the array. Compute the parent
        # group's path directly -- this works for local paths.
        full_path = getattr(ds, "_cellmap_path", None)
        if full_path:
            parent_path = os.path.dirname(str(full_path).rstrip("/"))
            if parent_path:
                try:
                    parent = zarr.open_group(parent_path, mode="r")
                    return dict(parent.attrs)
                except Exception:
                    return None
            return None

        # Get the filesystem path of the array's store root
        store = ds.data.store
        store_root = getattr(store, "root", None)
        if store_root:
            # store.root may be a pathlib.Path or a string (possibly with file:// prefix)
            store_root = str(store_root)
            if store_root.startswith("file://"):
                store_root = store_root[len("file://"):]

        # Determine the array's position within the store
        array_name = getattr(ds.data, "name", None) or getattr(ds.data, "path", "")
        array_name = array_name.strip("/")

        if array_name and store_root:
            # Array opened within a group hierarchy — navigate via store
            parent_path = "/".join(array_name.split("/")[:-1])
            if parent_path:
                parent = zarr.open_group(store, mode="r", path=parent_path)
                return dict(parent.attrs)
        elif store_root:
            # Array opened directly (zarr.open_array on full path) —
            # the store root IS the array path, so parent is one directory up
            parent_dir = os.path.dirname(store_root)
            if os.path.isdir(parent_dir):
                parent = zarr.open_group(parent_dir, mode="r")
                return dict(parent.attrs)
    except Exception:
        pass
    return None

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

    Precedence: the OME-NGFF ``multiscales`` scale at the opened level is the
    canonical source (it is the actual spec, and level-aware by construction).
    Funlib-style per-array ``voxel_size`` attrs are accepted as a legacy
    fallback. N5 ``pixelResolution`` and funlib's own parsed value follow.

    If OME and the per-array ``voxel_size`` attr are BOTH present and
    disagree, this raises ValueError -- silent precedence here is the exact
    failure mode that lets stale per-array attrs override corrected group
    metadata. Reconcile before downstream operations misread.

    Args:
        ds: A CellMapArray (returned by open_dataset).

    Returns:
        Tuple of floats representing the true voxel size.
    """
    attrs = dict(ds.data.attrs)
    scale_name = _array_scale_name(ds)

    ome_vs = _ome_scale_for(attrs, scale_name) or _ome_scale_for(
        _read_parent_attrs(ds) or {}, scale_name
    )
    arr_vs = (
        tuple(float(v) for v in attrs["voxel_size"])
        if "voxel_size" in attrs
        else None
    )

    if ome_vs is not None and arr_vs is not None and not _values_match(ome_vs, arr_vs):
        raise ValueError(
            _conflict_msg(
                ds,
                "voxel_size",
                "OME multiscales scale",
                ome_vs,
                "per-array voxel_size attr",
                arr_vs,
            )
        )

    if ome_vs is not None:
        return ome_vs
    if arr_vs is not None:
        return arr_vs

    # N5 pixelResolution (legacy per-array fallback)
    if "pixelResolution" in attrs:
        return tuple(float(v) for v in attrs["pixelResolution"]["dimensions"])

    # Last-ditch fallback to whatever funlib already parsed
    return tuple(float(v) for v in ds.voxel_size)


def read_raw_offset(ds):
    """Read the original float offset/translation from zarr/n5 attributes.

    Same precedence and conflict semantics as :func:`read_raw_voxel_size`:
    OME ``multiscales`` translation at the opened level is canonical, the
    per-array ``offset`` attr is legacy fallback, and a disagreement between
    the two raises ValueError.

    Args:
        ds: A CellMapArray (returned by open_dataset).

    Returns:
        Tuple of floats representing the stored offset/translation
        (per OME-NGFF: the voxel CENTER of voxel [0,0,0]).
    """
    attrs = dict(ds.data.attrs)
    scale_name = _array_scale_name(ds)

    ome_tr = _ome_translation_for(attrs, scale_name) or _ome_translation_for(
        _read_parent_attrs(ds) or {}, scale_name
    )
    arr_off = (
        tuple(float(v) for v in attrs["offset"]) if "offset" in attrs else None
    )

    if ome_tr is not None and arr_off is not None and not _values_match(ome_tr, arr_off):
        raise ValueError(
            _conflict_msg(
                ds,
                "translation/offset",
                "OME multiscales translation",
                ome_tr,
                "per-array offset attr",
                arr_off,
            )
        )

    if ome_tr is not None:
        return ome_tr
    if arr_off is not None:
        return arr_off

    return tuple(float(v) for v in ds.roi.offset)


def _ome_scale_for(attrs, scale_name):
    if not attrs or "multiscales" not in attrs:
        return None
    try:
        return _extract_ome_scale(attrs, scale_name)
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def _ome_translation_for(attrs, scale_name):
    if not attrs or "multiscales" not in attrs:
        return None
    try:
        return _extract_ome_translation(attrs, scale_name)
    except (KeyError, IndexError, TypeError):
        return None


def _values_match(a, b, atol=1e-6):
    """Element-wise float compare with absolute tolerance; length-aware."""
    if a is None or b is None:
        return True
    try:
        a = tuple(float(v) for v in a)
        b = tuple(float(v) for v in b)
    except (TypeError, ValueError):
        return False
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= atol for x, y in zip(a, b))


def _conflict_msg(ds, kind, name_a, val_a, name_b, val_b):
    return (
        f"Metadata conflict on {kind} at {_ds_path(ds)}:\n"
        f"  {name_a}: {list(val_a)}\n"
        f"  {name_b}: {list(val_b)}\n"
        f"Fix one to match the other (OME-NGFF convention: voxel CENTERS), "
        f"or delete the stale attr."
    )


def _ds_path(ds):
    """Best-effort path string for diagnostic messages."""
    try:
        store_root = getattr(getattr(ds.data, "store", None), "root", None)
        if store_root:
            s = str(store_root)
            if s.startswith("file://"):
                s = s[len("file://"):]
            return s.rstrip("/")
        return getattr(ds.data, "name", None) or getattr(ds.data, "path", "?")
    except Exception:
        return "?"


def _select_ome_dataset(attrs, scale_name=None):
    """Pick the multiscales ``datasets`` entry for the level being opened.

    Each scale level (s0, s1, ...) carries its OWN scale + translation, so we
    must match the opened array's leaf name (e.g. "s1") against the dataset
    ``path``. Falls back to the first entry when the level can't be identified
    (single-scale metadata, or a non-standard array name).
    """
    datasets = attrs["multiscales"][0]["datasets"]
    if scale_name is not None:
        for d in datasets:
            if d.get("path") == scale_name:
                return d
    return datasets[0]


def _extract_ome_scale(attrs, scale_name=None):
    """Extract voxel size from OME-Zarr multiscales metadata for a given level."""
    transforms = _select_ome_dataset(attrs, scale_name)["coordinateTransformations"]
    for t in transforms:
        if t["type"] == "scale":
            return tuple(float(v) for v in t["scale"])
    raise ValueError("No scale transform found in OME-Zarr multiscales metadata")


def _extract_ome_translation(attrs, scale_name=None):
    """Extract offset from OME-Zarr multiscales metadata for a given level."""
    transforms = _select_ome_dataset(attrs, scale_name)["coordinateTransformations"]
    for t in transforms:
        if t["type"] == "translation":
            return tuple(float(v) for v in t["translation"])
    return None


def _array_scale_name(ds):
    """Leaf name of the array's location (e.g. "s1"), used to select the
    matching OME multiscales dataset entry. Returns None if it can't be
    determined (then callers fall back to the first multiscales entry)."""
    try:
        import os

        # Prefer the path the caller used to open this array -- works for
        # both local paths and remote URIs (s3://...).
        full_path = getattr(ds, "_cellmap_path", None)
        if full_path:
            return os.path.basename(str(full_path).rstrip("/")) or None

        array_name = getattr(ds.data, "name", None) or getattr(ds.data, "path", "")
        array_name = str(array_name).strip("/")
        if array_name:
            return array_name.split("/")[-1]
        store_root = getattr(getattr(ds.data, "store", None), "root", None)
        if store_root:
            s = str(store_root)
            if s.startswith("file://"):
                s = s[len("file://"):]
            return os.path.basename(s.rstrip("/"))
    except Exception:
        pass
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

"""Neuroglancer precomputed volume readers.

Accepts paths of the form ``precomputed://<URL>`` where ``<URL>`` may be
``gs://bucket/path``, ``s3://bucket/path``, ``http(s)://...``, or a
local filesystem path. The ``precomputed://`` prefix is honored for
backward compatibility but is not required — :func:`_detect_zarr_driver`
probes for an ``info`` marker file and routes content-detected
precomputed volumes here automatically.

A trailing path segment may name a specific scale by its ``key`` in the
``info`` file (e.g. ``.../seg/s0`` or ``.../seg/8.0x8.0x8.0``);
otherwise scale 0 (highest resolution) is used.
"""
from __future__ import annotations

import logging
import posixpath

from cellmap_analyze.util.io_util import (
    kvstore_for_path,
    path_basename,
    path_dirname,
    path_join,
    read_json_path,
    strip_precomputed_prefix,
)

logger = logging.getLogger(__name__)


def _fetch_info(path):
    """Read the ``info`` JSON sitting at ``<path>/info``.

    Returns the parsed dict, or ``None`` if no info marker is present.
    Uses :func:`read_json_path`, which already handles every supported
    scheme (s3/gs/http(s)/local).
    """
    return read_json_path(path_join(path, "info"))


def parse_precomputed_path(path):
    """Resolve a precomputed path into (kvstore, base_path, info, scale_index).

    Tries to find the ``info`` file at ``path`` first, then at its
    parent (which covers paths like ``.../segmentation/s0`` where the
    trailing segment names a scale rather than a real subdirectory).

    Returns
    -------
    tuple[dict, str, dict, int]
        ``(kvstore_spec, base_path, info_dict, scale_index)``. The
        ``kvstore_spec`` is suitable for use as the ``kvstore`` field
        of a tensorstore spec (its ``path`` must still be set by the
        caller). ``base_path`` is the directory that holds ``info``.
    """
    inner = strip_precomputed_prefix(path).rstrip("/")
    kvstore, _key = kvstore_for_path(inner)

    info = _fetch_info(inner)
    base_path = inner
    scale_key = None

    if info is None:
        parent = path_dirname(inner)
        if parent and parent != inner:
            info = _fetch_info(parent)
            if info is not None:
                base_path = parent
                scale_key = path_basename(inner)

    if info is None:
        raise FileNotFoundError(
            f"No precomputed ``info`` file found at {inner!r} or its parent"
        )

    scales = info.get("scales", [])
    if not scales:
        raise ValueError(f"precomputed info at {base_path!r} has no scales")

    scale_index = 0
    if scale_key is not None:
        for i, s in enumerate(scales):
            if s.get("key") == scale_key:
                scale_index = i
                break
        else:
            raise ValueError(
                f"scale key {scale_key!r} not found in info "
                f"(available keys: {[s.get('key') for s in scales]})"
            )

    return kvstore, base_path, info, scale_index


def open_precomputed_tensorstore(path, scale_index=None):
    """Open a precomputed volume with TensorStore and drop the channel axis.

    Returns a 3D TensorStore handle in (x, y, z) order, matching the
    layout of an N5 dataset so the existing ``swap_axes=True`` codepath
    in :class:`ImageDataInterface` converts it to ZYX.
    """
    import tensorstore as ts

    _kvstore0, base_path, _info, default_scale = parse_precomputed_path(path)
    if scale_index is None:
        scale_index = default_scale

    # Build a kvstore spec rooted at the directory holding ``info``.
    # ``kvstore_for_path`` returns the kvstore with the path stripped
    # out as ``key``; tensorstore expects that key as ``path`` on the
    # kvstore (with a trailing slash so it's treated as a directory).
    kvstore, key = kvstore_for_path(base_path)
    kvstore["path"] = (key.rstrip("/") + "/") if key else ""

    spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": kvstore,
        "scale_index": int(scale_index),
    }
    ds = ts.open(spec, read=True, write=False).result()

    # Drop the channel dim so the dataset looks 3D in (x, y, z) order.
    if "channel" in ds.domain.labels:
        ds = ds[ts.d["channel"][0]]
    elif len(ds.domain) == 4:
        ds = ds[..., 0]
    # The precomputed driver bakes ``voxel_offset`` into the array
    # domain so voxel indices start at the offset, not at 0. The
    # cellmap-analyze read path (``to_ndarray_tensorstore``) handles
    # the physical offset itself and assumes a 0-based voxel array
    # (the N5 convention). Translate the domain to origin so the two
    # conventions agree.
    if any(o != 0 for o in ds.domain.origin):
        ds = ds.translate_to[(0,) * len(ds.domain)]
    return ds


def precomputed_array_metadata(path, scale_index=None):
    """Read shape/dtype/chunks/voxel_size/offset for a precomputed scale.

    All spatial values are returned in ZYX order to match the rest of
    the codebase. Voxel offset (in voxels, XYZ in the info file) is
    converted to physical units in ZYX. The returned ``offset`` is the
    CORNER of voxel [0,0,0] (i.e. ``voxel_offset * resolution``), which
    is the precomputed convention; callers that want the OME-NGFF
    center-translation convention must add ``vs/2``.
    """
    _, _, info, default_scale = parse_precomputed_path(path)
    if scale_index is None:
        scale_index = default_scale
    scale = info["scales"][scale_index]

    size_xyz = list(scale["size"])
    resolution_xyz = [float(v) for v in scale["resolution"]]
    voxel_offset_xyz = list(scale.get("voxel_offset") or [0, 0, 0])
    chunk_sizes = scale.get("chunk_sizes") or [[64, 64, 64]]
    chunk_xyz = list(chunk_sizes[0])
    dtype = info.get("data_type", "uint64")

    shape_zyx = tuple(size_xyz[::-1])
    chunk_zyx = tuple(chunk_xyz[::-1])
    voxel_size_zyx = [float(v) for v in resolution_xyz[::-1]]
    offset_zyx = [
        float(o) * float(r)
        for o, r in zip(voxel_offset_xyz[::-1], resolution_xyz[::-1])
    ]
    return {
        "shape": shape_zyx,
        "chunks": chunk_zyx,
        "dtype": dtype,
        "voxel_size": voxel_size_zyx,
        "offset_corner": offset_zyx,
        "scale_index": scale_index,
        "info": info,
    }

"""Hand-rolled writer for the neuroglancer_uint64_sharded_v1 skeleton format.

Spec reference:
https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/sharded.md

Only the subset we need is supported: identity hash, raw data/index encoding.
Reading is left to neuroglancer; this module only packs already-encoded
per-ID skeleton bytes into shard files plus the sharding spec for the info file.
"""

from __future__ import annotations

import math
import os
import struct
from typing import Iterable


SHARDING_TYPE = "neuroglancer_uint64_sharded_v1"


def make_sharding_spec(
    preshift_bits: int = 0, shard_bits: int = 1, minishard_bits: int = 6
) -> dict:
    """Return the sharding spec dict for an ``info`` file. The same
    parameters must be passed to :func:`pack_sharded_skeletons` so the
    written shards match what readers expect.
    """
    return {
        "@type": SHARDING_TYPE,
        "preshift_bits": preshift_bits,
        "hash": "identity",
        "minishard_bits": minishard_bits,
        "shard_bits": shard_bits,
        "minishard_index_encoding": "raw",
        "data_encoding": "raw",
    }


def _assign_bucket(
    object_id: int, preshift_bits: int, shard_bits: int, minishard_bits: int
) -> tuple[int, int]:
    """Map a chunk ID to (shard_no, minishard_no) per the sharded v1 spec.

    With hash="identity":
        hashed = object_id >> preshift_bits
        minishard = hashed & ((1 << minishard_bits) - 1)
        shard     = (hashed >> minishard_bits) & ((1 << shard_bits) - 1)
    """
    hashed = object_id >> preshift_bits
    minishard = hashed & ((1 << minishard_bits) - 1)
    shard = (hashed >> minishard_bits) & ((1 << shard_bits) - 1)
    return shard, minishard


def _shard_filename(shard_no: int, shard_bits: int) -> str:
    # Spec: lowercase hex, zero-padded to ceil(shard_bits/4) digits. We refuse
    # shard_bits=0 in pack_sharded_skeletons so this is always >=1, matching
    # the spec exactly and avoiding the POSIX-hidden ".shard" filename.
    digits = math.ceil(shard_bits / 4)
    return f"{shard_no:0{digits}x}.shard"


def _pack_minishard(
    chunks: list[tuple[int, bytes]], cursor: int
) -> tuple[bytes, bytes, int]:
    """Build the data blob + minishard index for a sorted list of chunks.

    Returns (data_blob, index_blob, new_cursor). All offsets in the index are
    relative to the end of the shard index (i.e., the same "post-shard-index"
    coordinate system in which `cursor` lives).
    """
    n = len(chunks)
    ids = [c[0] for c in chunks]
    sizes = [len(c[1]) for c in chunks]

    # array[0] — chunk IDs, delta-encoded: first is absolute, rest are diffs.
    id_deltas = [ids[0]] + [ids[i] - ids[i - 1] for i in range(1, n)]

    # array[1] — chunk start offsets, delta-encoded relative to *end of previous chunk*.
    # First is absolute (cursor at start of this minishard's data).
    # We pack chunks contiguously, so deltas for i>=1 are all zero.
    offset_deltas = [cursor] + [0] * (n - 1)

    # array[2] — chunk sizes, literal.
    minishard_index = struct.pack(
        f"<{3 * n}Q", *id_deltas, *offset_deltas, *sizes
    )

    data_blob = b"".join(c[1] for c in chunks)
    new_cursor = cursor + len(data_blob) + len(minishard_index)
    return data_blob, minishard_index, new_cursor


def pack_sharded_skeletons(
    id_to_bytes: dict[int, bytes],
    output_dir: str,
    preshift_bits: int = 0,
    shard_bits: int = 1,
    minishard_bits: int = 6,
) -> dict:
    """Pack ID→skeleton-bytes into precomputed sharded v1 shard files.

    Writes ``<output_dir>/<shard_hex>.shard`` files. Returns the sharding spec
    dict to embed under the "sharding" key in the ``info`` file.

    Only ``hash="identity"`` and raw (uncompressed) encoding are supported;
    MWS-style densely-numbered uint64 IDs spread evenly under identity hashing.
    """
    if shard_bits < 1:
        raise ValueError(
            "shard_bits must be >= 1; shard_bits=0 would emit a hidden "
            "'.shard' file per the spec, which is fragile on POSIX."
        )
    if minishard_bits < 0:
        raise ValueError("minishard_bits must be >= 0")
    os.makedirs(output_dir, exist_ok=True)
    n_minishards_per_shard = 1 << minishard_bits
    n_shards = 1 << shard_bits

    # Bucket per (shard, minishard).
    buckets: dict[tuple[int, int], list[tuple[int, bytes]]] = {}
    for obj_id, data in id_to_bytes.items():
        s, m = _assign_bucket(int(obj_id), preshift_bits, shard_bits, minishard_bits)
        buckets.setdefault((s, m), []).append((int(obj_id), data))

    for shard_no in range(n_shards):
        # For each minishard slot, build its (data, index) bytes and track
        # the start/end offsets of its index relative to end-of-shard-index.
        shard_index_entries: list[tuple[int, int]] = []  # (start, end) per minishard
        body_parts: list[bytes] = []
        cursor = 0  # bytes written after the shard index header

        for mini_no in range(n_minishards_per_shard):
            chunks = sorted(buckets.get((shard_no, mini_no), []), key=lambda c: c[0])
            if not chunks:
                # Empty minishard: zero-length index at the current cursor.
                shard_index_entries.append((cursor, cursor))
                continue

            data_blob, index_blob, new_cursor = _pack_minishard(chunks, cursor)
            body_parts.append(data_blob)
            data_end = cursor + len(data_blob)
            shard_index_entries.append((data_end, data_end + len(index_blob)))
            body_parts.append(index_blob)
            cursor = new_cursor

        shard_index_buf = b"".join(
            struct.pack("<QQ", s, e) for s, e in shard_index_entries
        )
        shard_path = os.path.join(output_dir, _shard_filename(shard_no, shard_bits))
        with open(shard_path, "wb") as f:
            f.write(shard_index_buf)
            for part in body_parts:
                f.write(part)

    return make_sharding_spec(preshift_bits, shard_bits, minishard_bits)


def read_chunk_from_shard(
    shard_dir: str,
    object_id: int,
    sharding: dict,
) -> bytes | None:
    """Read a single chunk back out of the shard files. Used for tests.

    Returns the chunk bytes, or None if the chunk is absent.
    """
    preshift_bits = sharding["preshift_bits"]
    shard_bits = sharding["shard_bits"]
    minishard_bits = sharding["minishard_bits"]
    if sharding.get("hash", "identity") != "identity":
        raise NotImplementedError("Only identity hashing is supported")
    if sharding.get("data_encoding", "raw") != "raw":
        raise NotImplementedError("Only raw data_encoding is supported")
    if sharding.get("minishard_index_encoding", "raw") != "raw":
        raise NotImplementedError("Only raw minishard_index_encoding is supported")

    shard_no, mini_no = _assign_bucket(
        int(object_id), preshift_bits, shard_bits, minishard_bits
    )
    shard_path = os.path.join(shard_dir, _shard_filename(shard_no, shard_bits))
    if not os.path.exists(shard_path):
        return None

    n_minishards = 1 << minishard_bits
    shard_index_size = 16 * n_minishards
    with open(shard_path, "rb") as f:
        header = f.read(shard_index_size)
        start, end = struct.unpack_from("<QQ", header, mini_no * 16)
        if end <= start:
            return None
        f.seek(shard_index_size + start)
        raw = f.read(end - start)
        n = (end - start) // 24
        triples = struct.unpack(f"<{3 * n}Q", raw)
        id_deltas = triples[0:n]
        offset_deltas = triples[n : 2 * n]
        sizes = triples[2 * n : 3 * n]

        ids = []
        last = 0
        for d in id_deltas:
            last += d
            ids.append(last)

        # offset[i] = shard_index_end + sum(offset_deltas[0..i]) + sum(sizes[0..i-1])
        target = int(object_id)
        if target not in ids:
            return None
        i = ids.index(target)
        offset = shard_index_size + sum(offset_deltas[: i + 1]) + sum(sizes[:i])
        f.seek(offset)
        return f.read(sizes[i])

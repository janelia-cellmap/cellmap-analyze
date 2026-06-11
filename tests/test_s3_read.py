"""Read-only S3 support: end-to-end test against an in-process mock S3.

Uses moto's ThreadedMotoServer so both s3fs (which zarr uses for ``s3://``
URIs) and tensorstore (which has its own HTTP client) talk to the same mock
endpoint. No real network, no AWS credentials needed. ``s3fs`` and
``moto[s3,server]`` are in the ``[dev]`` extra.

Coverage:
- IDI accepts ``s3://`` URIs and reports the right voxel_size/translation.
- ``read_raw_voxel_size`` / ``read_raw_offset`` correctly walk to the parent
  group's OME multiscales when only that is present (per the read-precedence
  rules in PR #73; this is the standard layout for cellmap data on S3).
- ``to_ndarray_ts`` reads voxel data through tensorstore's S3 driver and
  returns the same bytes that were uploaded.
- The "I didn't change anything else" guarantee: ``_array_scale_name`` /
  multiscale level selection still works when the path is an s3:// URI
  (i.e. opening ``s3://.../seg/s1`` picks the s1 transform, not s0's).
"""
from __future__ import annotations

import json
import socket

import numpy as np
import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def s3_server():
    """Start an in-process mock S3 server. All s3:// reads in tests
    using this fixture talk to it via ``AWS_ENDPOINT_URL``."""
    from moto.server import ThreadedMotoServer

    port = _free_port()
    server = ThreadedMotoServer(port=port, ip_address="127.0.0.1")
    server.start()
    yield f"http://127.0.0.1:{port}"
    server.stop()


@pytest.fixture
def s3_env(monkeypatch, s3_server):
    """Point boto3/s3fs/tensorstore at the moto server."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ENDPOINT_URL", s3_server)
    return s3_server


def _make_bucket(s3_env, bucket: str):
    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_env,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    s3.create_bucket(Bucket=bucket)


def _open_group_on_mock_s3(s3_env, url, mode):
    """Open a zarr group at ``url`` on the moto mock S3 server.

    Goes through zarr's URL-based store creation (with ``storage_options``)
    rather than manually instantiating an s3fs filesystem -- the manual
    path tripped zarr 3.x's ``async_impl`` check on some s3fs versions.
    """
    import zarr

    return zarr.open_group(
        url,
        mode=mode,
        zarr_format=3,
        storage_options={
            "client_kwargs": {"endpoint_url": s3_env, "region_name": "us-east-1"},
            "key": "test",
            "secret": "test",
        },
    )


def _write_zarr_dataset(s3_env, bucket, group_path, level_name, vs, trans, data):
    """Upload a single-level OME-NGFF zarr v3 dataset to mock S3.

    Group (parent) holds the multiscales metadata; the array at
    ``group_path/level_name`` holds the voxel data. Mirrors how external
    OME-NGFF datasets are typically laid out (per-level transforms only in
    the parent group, no per-array offset/voxel_size attrs).
    """
    group = _open_group_on_mock_s3(s3_env, f"s3://{bucket}/{group_path}", mode="w")
    group.attrs["multiscales"] = [
        {
            "axes": [
                {"name": a, "type": "space", "unit": "nanometer"}
                for a in ("z", "y", "x")
            ],
            "datasets": [
                {
                    "path": level_name,
                    "coordinateTransformations": [
                        {"scale": list(vs), "type": "scale"},
                        {"translation": list(trans), "type": "translation"},
                    ],
                }
            ],
            "name": "",
            "version": "0.4",
        }
    ]

    # The data array
    arr = group.create_array(
        name=level_name,
        shape=data.shape,
        chunks=data.shape,
        dtype=data.dtype,
    )
    arr[:] = data


def test_s3_read_zarr_via_idi(s3_env):
    """Open s3://.../seg/s0 via IDI; voxel_size, offset, and data all
    come back correctly through the OME multiscales metadata."""
    from cellmap_analyze.util.image_data_interface import ImageDataInterface

    bucket = "cellmap-test"
    _make_bucket(s3_env, bucket)

    # OME pyramid-like: vs=8, translation=4 → OME-correct s1 of a base-8
    # pyramid would actually be (V-V_base)/2 = 0, but a non-zero
    # translation exercises the center<->corner conversion (PR #70).
    data = np.arange(8 * 8 * 8, dtype=np.uint8).reshape((8, 8, 8))
    _write_zarr_dataset(
        s3_env,
        bucket,
        "data.zarr/seg",
        "s0",
        vs=(8.0, 8.0, 8.0),
        trans=(4.0, 4.0, 4.0),
        data=data,
    )

    uri = f"s3://{bucket}/data.zarr/seg/s0"
    idi = ImageDataInterface(uri)

    # Metadata round-trips through the parent group's OME multiscales.
    assert tuple(idi.voxel_size) == (8, 8, 8)
    # OME translation = voxel CENTER; v0.3.0 internally exposes the corner.
    # trans=4, vs=8 → corner = 4 − 8/2 = 0.
    assert tuple(idi.offset) == (0, 0, 0)

    # And the actual voxel data comes back through tensorstore's s3 driver.
    read = idi.to_ndarray_ts(idi.roi)
    np.testing.assert_array_equal(read, data)


def test_s3_read_picks_correct_multiscale_level(s3_env):
    """The multiscale-level fix (PR #73) also has to work over s3:// --
    opening s1 must pick the s1 transform from the parent's multiscales
    metadata, not s0's."""
    from cellmap_analyze.util.image_data_interface import ImageDataInterface

    bucket = "cellmap-test-multiscale"
    _make_bucket(s3_env, bucket)

    # Build a two-level group: s0 vs=4 trans=0; s1 vs=8 trans=2.
    group = _open_group_on_mock_s3(
        s3_env, f"s3://{bucket}/data.zarr/seg", mode="w"
    )
    group.attrs["multiscales"] = [
        {
            "axes": [
                {"name": a, "type": "space", "unit": "nanometer"}
                for a in ("z", "y", "x")
            ],
            "datasets": [
                {
                    "path": "s0",
                    "coordinateTransformations": [
                        {"scale": [4.0, 4.0, 4.0], "type": "scale"},
                        {"translation": [0.0, 0.0, 0.0], "type": "translation"},
                    ],
                },
                {
                    "path": "s1",
                    "coordinateTransformations": [
                        {"scale": [8.0, 8.0, 8.0], "type": "scale"},
                        {"translation": [2.0, 2.0, 2.0], "type": "translation"},
                    ],
                },
            ],
            "name": "",
            "version": "0.4",
        }
    ]
    for level, shape, val in [("s0", (16, 16, 16), 1), ("s1", (8, 8, 8), 2)]:
        a = group.create_array(name=level, shape=shape, chunks=shape, dtype="uint8")
        a[:] = val

    # s0: vs=4, corner = 0 − 4/2 = −2.
    idi0 = ImageDataInterface(f"s3://{bucket}/data.zarr/seg/s0")
    assert tuple(idi0.voxel_size) == (4, 4, 4)
    assert tuple(idi0.offset) == (-2, -2, -2)
    assert np.all(idi0.to_ndarray_ts(idi0.roi) == 1)

    # s1: vs=8, trans=2 → corner = 2 − 8/2 = −2. Same corner as s0
    # (the OME pyramid invariant) -- the LEVEL is what we have to pick
    # correctly here.
    idi1 = ImageDataInterface(f"s3://{bucket}/data.zarr/seg/s1")
    assert tuple(idi1.voxel_size) == (8, 8, 8), (
        f"expected s1 voxel_size 8 but got {tuple(idi1.voxel_size)} -- "
        f"this is the datasets[0]-hardcoding bug if it fails"
    )
    assert tuple(idi1.offset) == (-2, -2, -2)
    assert np.all(idi1.to_ndarray_ts(idi1.roi) == 2)

"""Read-only support for ``s3://`` zarr datasets, tested end-to-end against
an in-process mock S3 (moto). No fsspec, no s3fs -- the test path mirrors
the production path: zarr metadata via stdlib ``urllib`` (with
``AWS_ENDPOINT_URL`` redirecting to moto), voxel data via tensorstore's
native S3 driver pointed at the same endpoint.

Coverage:
- IDI accepts ``s3://`` URIs and reports the right voxel_size /
  translation (which lives in the parent group's OME multiscales).
- ``to_ndarray_ts`` reads voxel data through tensorstore's S3 driver and
  returns the same bytes that were uploaded.
- Multiscale level selection works over S3 (PR #73's ``datasets[0]``
  fix continuing to work when metadata is fetched over the wire).
"""
from __future__ import annotations

import os
import socket

import numpy as np
import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def s3_server():
    """Start an in-process mock S3 server."""
    from moto.server import ThreadedMotoServer

    port = _free_port()
    server = ThreadedMotoServer(port=port, ip_address="127.0.0.1")
    server.start()
    yield f"http://127.0.0.1:{port}"
    server.stop()


@pytest.fixture
def s3_env(monkeypatch, s3_server):
    """Point boto3 / tensorstore / our own ``url_to_public_https`` at moto."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ENDPOINT_URL", s3_server)
    return s3_server


def _boto_client(s3_env):
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=s3_env,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


def _make_bucket(s3_env, bucket: str):
    """Create a public-read bucket on the mock S3 server.

    cellmap-analyze reads zarr metadata via stdlib ``urllib.request.urlopen``
    against the bucket's HTTPS URL -- unsigned. Real cellmap data on S3 is
    public-read, which is what makes that unsigned read work. We mirror
    that here by setting a public-read GetObject policy on the bucket so
    the test exercises the same unsigned-read code path.
    """
    import json as _json

    s3 = _boto_client(s3_env)
    s3.create_bucket(Bucket=bucket)
    s3.put_bucket_policy(
        Bucket=bucket,
        Policy=_json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "PublicRead",
                        "Effect": "Allow",
                        "Principal": "*",
                        "Action": "s3:GetObject",
                        "Resource": f"arn:aws:s3:::{bucket}/*",
                    }
                ],
            }
        ),
    )


def _upload_dir_to_s3(s3_env, local_root: str, bucket: str, key_prefix: str):
    """Upload every file under ``local_root`` to ``s3://bucket/key_prefix/``
    preserving relative paths. Used to mirror a locally-built zarr to
    mock S3 without needing an async/fsspec stack to write directly."""
    s3 = _boto_client(s3_env)
    key_prefix = key_prefix.strip("/")
    for dirpath, _dirnames, filenames in os.walk(local_root):
        for name in filenames:
            local_path = os.path.join(dirpath, name)
            rel = os.path.relpath(local_path, local_root).replace(os.sep, "/")
            key = f"{key_prefix}/{rel}".lstrip("/")
            with open(local_path, "rb") as f:
                s3.put_object(Bucket=bucket, Key=key, Body=f.read())


def _build_local_ome_zarr(
    root: str, level_data: dict[str, tuple[tuple, np.ndarray, tuple, tuple]],
):
    """Build a v3 OME-Zarr group at ``root`` with one or more levels.

    ``level_data`` is ``{level_name: (vs_tuple, data_ndarray, vs, trans)}``
    -- we accept ``vs``/``trans`` both at the level entry (for the OME
    multiscales) and as the first tuple element (for the array's own
    voxel_size convention). Locally-built and then mirrored to S3 by the
    caller -- this lets the test code stay stdlib-only (no s3fs).
    """
    import zarr

    group = zarr.open_group(root, mode="w", zarr_format=3)
    datasets = []
    for name, (_, data, vs, trans) in level_data.items():
        datasets.append(
            {
                "path": name,
                "coordinateTransformations": [
                    {"scale": list(vs), "type": "scale"},
                    {"translation": list(trans), "type": "translation"},
                ],
            }
        )
        arr = group.create_array(
            name=name, shape=data.shape, chunks=data.shape, dtype=data.dtype
        )
        arr[:] = data
    group.attrs["multiscales"] = [
        {
            "axes": [
                {"name": a, "type": "space", "unit": "nanometer"}
                for a in ("z", "y", "x")
            ],
            "datasets": datasets,
            "name": "",
            "version": "0.4",
        }
    ]


def test_s3_read_zarr_via_idi(tmp_path, s3_env):
    """Open s3://.../seg/s0 via IDI; voxel_size, offset, and data come
    back correctly through the parent group's OME multiscales."""
    from cellmap_analyze.util.image_data_interface import ImageDataInterface

    bucket = "cellmap-test"
    _make_bucket(s3_env, bucket)

    data = np.arange(8 * 8 * 8, dtype=np.uint8).reshape((8, 8, 8))
    local_root = tmp_path / "local_data.zarr" / "seg"
    _build_local_ome_zarr(
        str(local_root),
        {"s0": (None, data, (8.0, 8.0, 8.0), (4.0, 4.0, 4.0))},
    )
    _upload_dir_to_s3(s3_env, str(local_root), bucket, "data.zarr/seg")

    idi = ImageDataInterface(f"s3://{bucket}/data.zarr/seg/s0")

    # Metadata round-trips through the parent group's OME multiscales.
    assert tuple(idi.voxel_size) == (8, 8, 8)
    # OME translation = voxel CENTER; v0.3.0 internally exposes the corner.
    # trans=4, vs=8 → corner = 4 − 8/2 = 0.
    assert tuple(idi.offset) == (0, 0, 0)

    # And the actual voxel data comes back through tensorstore's s3 driver.
    read = idi.to_ndarray_ts(idi.roi)
    np.testing.assert_array_equal(read, data)


def test_s3_read_picks_correct_multiscale_level(tmp_path, s3_env):
    """The level-selection fix (PR #73) also has to work over s3:// --
    opening s1 picks the s1 transform from the parent's multiscales
    metadata, not s0's."""
    from cellmap_analyze.util.image_data_interface import ImageDataInterface

    bucket = "cellmap-test-multiscale"
    _make_bucket(s3_env, bucket)

    s0 = np.ones((16, 16, 16), dtype=np.uint8) * 1
    s1 = np.ones((8, 8, 8), dtype=np.uint8) * 2
    local_root = tmp_path / "ms.zarr" / "seg"
    _build_local_ome_zarr(
        str(local_root),
        {
            "s0": (None, s0, (4.0, 4.0, 4.0), (0.0, 0.0, 0.0)),
            "s1": (None, s1, (8.0, 8.0, 8.0), (2.0, 2.0, 2.0)),
        },
    )
    _upload_dir_to_s3(s3_env, str(local_root), bucket, "data.zarr/seg")

    # s0: vs=4, corner = 0 − 4/2 = −2.
    idi0 = ImageDataInterface(f"s3://{bucket}/data.zarr/seg/s0")
    assert tuple(idi0.voxel_size) == (4, 4, 4)
    assert tuple(idi0.offset) == (-2, -2, -2)
    assert np.all(idi0.to_ndarray_ts(idi0.roi) == 1)

    # s1: vs=8, trans=2 → corner = 2 − 8/2 = −2. Same corner as s0 (the
    # OME pyramid invariant) -- the LEVEL is what we have to pick
    # correctly here.
    idi1 = ImageDataInterface(f"s3://{bucket}/data.zarr/seg/s1")
    assert tuple(idi1.voxel_size) == (8, 8, 8), (
        f"expected s1 voxel_size 8 but got {tuple(idi1.voxel_size)} -- "
        f"this is the datasets[0]-hardcoding regression if it fails"
    )
    assert tuple(idi1.offset) == (-2, -2, -2)
    assert np.all(idi1.to_ndarray_ts(idi1.roi) == 2)

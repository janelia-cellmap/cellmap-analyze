"""Read-only support for neuroglancer precomputed volumes.

Covers both local-filesystem and S3 (via in-process moto) reads, the
``precomputed://`` URL prefix, and selecting a specific scale by the
``/<key>`` URL suffix. Mirrors ``test_s3_read.py``'s moto pattern.
"""
from __future__ import annotations

import json
import os
import shutil
import socket

import numpy as np
import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def s3_server():
    from moto.server import ThreadedMotoServer

    port = _free_port()
    server = ThreadedMotoServer(port=port, ip_address="127.0.0.1")
    server.start()
    yield f"http://127.0.0.1:{port}"
    server.stop()


@pytest.fixture
def s3_env(monkeypatch, s3_server):
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


def _make_public_bucket(s3_env, bucket: str):
    s3 = _boto_client(s3_env)
    s3.create_bucket(Bucket=bucket)
    s3.put_bucket_policy(
        Bucket=bucket,
        Policy=json.dumps(
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
    s3 = _boto_client(s3_env)
    key_prefix = key_prefix.strip("/")
    for dirpath, _dn, filenames in os.walk(local_root):
        for name in filenames:
            local_path = os.path.join(dirpath, name)
            rel = os.path.relpath(local_path, local_root).replace(os.sep, "/")
            key = f"{key_prefix}/{rel}".lstrip("/")
            with open(local_path, "rb") as f:
                s3.put_object(Bucket=bucket, Key=key, Body=f.read())


def _build_precomputed_volume(root: str, data_zyx: np.ndarray, voxel_offset_xyz, resolution_xyz):
    """Author a 1-scale precomputed volume on disk via tensorstore."""
    import tensorstore as ts

    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    size_xyz = list(data_zyx.shape[::-1])
    chunk_xyz = list(size_xyz)  # one-chunk-per-volume keeps the fixture tiny
    spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": {"driver": "file", "path": root + "/"},
        "multiscale_metadata": {
            "data_type": str(data_zyx.dtype),
            "num_channels": 1,
            "type": "segmentation",
        },
        "scale_metadata": {
            "size": size_xyz,
            "resolution": list(resolution_xyz),
            "voxel_offset": list(voxel_offset_xyz),
            "encoding": "raw",
            "chunk_size": chunk_xyz,
            "key": "s0",
        },
        "create": True,
    }
    ds = ts.open(spec).result()
    ds[...] = data_zyx.transpose(2, 1, 0)[..., None]


def test_precomputed_read_local(tmp_path):
    """Open a local precomputed volume; voxel_size + corner offset come
    from the ``info`` file, data round-trips through tensorstore."""
    from cellmap_analyze.util.image_data_interface import ImageDataInterface

    data = np.arange(4 * 6 * 8, dtype=np.uint8).reshape(4, 6, 8)  # ZYX
    root = str(tmp_path / "vol")
    _build_precomputed_volume(
        root, data, voxel_offset_xyz=[10, 20, 30], resolution_xyz=[8.0, 8.0, 8.0]
    )

    idi = ImageDataInterface(root)
    assert idi.filetype == "neuroglancer_precomputed"
    assert tuple(idi.voxel_size) == (8, 8, 8)
    # voxel_offset = (10, 20, 30) XYZ → ZYX corner = (30, 20, 10) * 8 = (240, 160, 80)
    assert tuple(idi.offset) == (240, 160, 80)
    np.testing.assert_array_equal(idi.to_ndarray_ts(idi.roi), data)


def test_precomputed_read_with_scale_suffix(tmp_path):
    """``.../vol/s0`` selects scale ``s0`` from the info file."""
    from cellmap_analyze.util.image_data_interface import ImageDataInterface

    data = np.arange(4 * 6 * 8, dtype=np.uint8).reshape(4, 6, 8)
    root = str(tmp_path / "vol")
    _build_precomputed_volume(root, data, [0, 0, 0], [4.0, 4.0, 4.0])

    idi = ImageDataInterface(root + "/s0")
    assert tuple(idi.voxel_size) == (4, 4, 4)
    np.testing.assert_array_equal(idi.to_ndarray_ts(idi.roi), data)


def test_precomputed_read_with_explicit_prefix(tmp_path):
    """``precomputed://`` prefix is accepted (neuroglancer-style URLs)."""
    from cellmap_analyze.util.image_data_interface import ImageDataInterface

    data = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    root = str(tmp_path / "vol")
    _build_precomputed_volume(root, data, [0, 0, 0], [16.0, 16.0, 16.0])

    idi = ImageDataInterface("precomputed://" + root)
    assert tuple(idi.voxel_size) == (16, 16, 16)
    np.testing.assert_array_equal(idi.to_ndarray_ts(idi.roi), data)


def test_precomputed_read_s3(tmp_path, s3_env):
    """Read a precomputed volume from mock S3, exercising the same code
    paths a real public-bucket read would: ``info`` via stdlib urllib,
    chunks via tensorstore's S3 driver. Mirrors test_s3_read.py."""
    from cellmap_analyze.util.image_data_interface import ImageDataInterface

    bucket = "cellmap-precomp"
    _make_public_bucket(s3_env, bucket)

    data = np.arange(4 * 6 * 8, dtype=np.uint8).reshape(4, 6, 8)
    local_root = str(tmp_path / "vol")
    _build_precomputed_volume(
        local_root, data, voxel_offset_xyz=[10, 20, 30], resolution_xyz=[8.0, 8.0, 8.0]
    )
    _upload_dir_to_s3(s3_env, local_root, bucket, "data/seg")

    idi = ImageDataInterface(f"s3://{bucket}/data/seg")
    assert idi.filetype == "neuroglancer_precomputed"
    assert tuple(idi.voxel_size) == (8, 8, 8)
    assert tuple(idi.offset) == (240, 160, 80)
    np.testing.assert_array_equal(idi.to_ndarray_ts(idi.roi), data)

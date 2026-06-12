"""``kvstore_for_path`` defaults s3:// reads to anonymous mode when no
AWS credentials are configured -- this is what makes public-bucket
reads (e.g. OpenOrganelle) fast instead of paying multi-second
IMDS/credential-chain timeouts inside tensorstore.
"""
from __future__ import annotations

import pytest

from cellmap_analyze.util.io_util import kvstore_for_path

_CRED_VARS = ("AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_ENDPOINT_URL")


@pytest.fixture
def no_aws_env(monkeypatch):
    for k in _CRED_VARS:
        monkeypatch.delenv(k, raising=False)


def test_s3_default_anonymous_when_no_creds(no_aws_env):
    """No AWS env vars → kvstore gets aws_credentials={type: anonymous}."""
    kv, key = kvstore_for_path("s3://janelia-cosem-datasets/jrc_hela-2/foo.zarr")
    assert kv["driver"] == "s3"
    assert kv["bucket"] == "janelia-cosem-datasets"
    assert kv.get("aws_credentials") == {"type": "anonymous"}
    assert key == "jrc_hela-2/foo.zarr"


@pytest.mark.parametrize(
    "var,value",
    [
        ("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE"),
        ("AWS_PROFILE", "default"),
        ("AWS_ENDPOINT_URL", "http://localhost:9000"),
    ],
)
def test_s3_honors_creds_env_var(no_aws_env, monkeypatch, var, value):
    """Any of the credential-related env vars switches off anonymous mode
    so the tensorstore default credential chain is used (or the moto
    endpoint, etc)."""
    monkeypatch.setenv(var, value)
    kv, _ = kvstore_for_path("s3://bucket/key")
    assert "aws_credentials" not in kv


def test_endpoint_url_still_attached(no_aws_env, monkeypatch):
    """When AWS_ENDPOINT_URL is set (moto / MinIO), the kvstore should
    pick it up AND skip anonymous mode."""
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9000")
    kv, _ = kvstore_for_path("s3://bucket/key")
    assert kv["endpoint"] == "http://127.0.0.1:9000"
    assert "aws_credentials" not in kv


def test_non_s3_schemes_unaffected(no_aws_env):
    """gs / http / file kvstores don't get aws_credentials injected."""
    for url in (
        "gs://bucket/key",
        "https://example.com/key",
        "file:///tmp/foo",
        "/tmp/foo",
    ):
        kv, _ = kvstore_for_path(url)
        assert "aws_credentials" not in kv

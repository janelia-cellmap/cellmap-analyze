import os

import dask
import pytest

from cellmap_analyze.util.dask_util import set_local_directory


@pytest.fixture
def clear_local_directory():
    """Ensure the jobqueue local-directory config starts and ends unset so the
    selection logic actually runs (it short-circuits when already set)."""
    key = "jobqueue.lsf.local-directory"
    dask.config.set({key: None})
    yield
    dask.config.set({key: None})


def test_skips_existing_but_unwritable_dir(tmp_path, clear_local_directory):
    """A directory that exists but isn't writable must be skipped in favor of
    the next writable candidate -- the bug was makedirs(exist_ok=True)
    succeeding on an unwritable dir and committing to it."""
    scratch = tmp_path / "scratch"
    fallback = tmp_path / "tmp"
    scratch.mkdir()
    scratch.chmod(0o500)  # r-x: exists but not writable
    try:
        set_local_directory("lsf", candidate_dirs=[str(scratch), str(fallback)])
        assert dask.config.get("jobqueue.lsf.local-directory") == str(fallback)
    finally:
        scratch.chmod(0o700)


def test_picks_first_writable_dir(tmp_path, clear_local_directory):
    first = tmp_path / "first"
    set_local_directory(
        "lsf", candidate_dirs=[str(first), str(tmp_path / "second")]
    )
    assert dask.config.get("jobqueue.lsf.local-directory") == str(first)
    assert os.path.isdir(first)


def test_raises_when_no_writable_dir(tmp_path, clear_local_directory):
    unwritable = tmp_path / "ro"
    unwritable.mkdir()
    unwritable.chmod(0o500)
    try:
        with pytest.raises(RuntimeError, match="writable local-directory"):
            set_local_directory("lsf", candidate_dirs=[str(unwritable)])
    finally:
        unwritable.chmod(0o700)

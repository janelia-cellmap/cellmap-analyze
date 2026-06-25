import zarr
import pytest

from cellmap_analyze.util.zarr_io import (
    list_multiscale_levels,
    resolve_scale_path,
)
from cellmap_analyze.util.image_data_interface import ImageDataInterface


def _make_multiscale_group(path, levels):
    """Create a minimal OME-NGFF multiscale group with isotropic levels.

    ``levels`` is a list of (level_name, voxel_size_float).
    """
    g = zarr.open_group(path, mode="w")
    for name, _ in levels:
        g.create_dataset(name, shape=(10, 10, 10), dtype="uint8")
    g.attrs.update(
        {
            "multiscales": [
                {
                    "axes": [
                        {"name": a, "type": "space", "unit": "nanometer"}
                        for a in "zyx"
                    ],
                    "datasets": [
                        {
                            "path": name,
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [vs, vs, vs]},
                                {"type": "translation", "translation": [0, 0, 0]},
                            ],
                        }
                        for name, vs in levels
                    ],
                    "version": "0.4",
                }
            ]
        }
    )
    return path


@pytest.fixture
def multiscale_group(tmp_path):
    path = str(tmp_path / "raw.zarr" / "em")
    return _make_multiscale_group(
        path, [("s0", 4.0), ("s1", 8.0), ("s2", 16.0)]
    )


def test_list_multiscale_levels(multiscale_group):
    levels = list_multiscale_levels(multiscale_group)
    assert levels == [
        ("s0", (4.0, 4.0, 4.0)),
        ("s1", (8.0, 8.0, 8.0)),
        ("s2", (16.0, 16.0, 16.0)),
    ]


def test_list_levels_returns_none_for_array(multiscale_group):
    # A concrete scale-level array is not a multiscale group.
    assert list_multiscale_levels(multiscale_group + "/s0") is None


def test_resolve_defaults_to_finest_scale(multiscale_group):
    assert resolve_scale_path(multiscale_group).endswith("/s0")


def test_resolve_matches_exact_target(multiscale_group):
    assert resolve_scale_path(
        multiscale_group, target_voxel_size=(8.0, 8.0, 8.0)
    ).endswith("/s1")


def test_resolve_matches_closest_target(multiscale_group):
    # 15nm is closest to the 16nm level.
    assert resolve_scale_path(
        multiscale_group, target_voxel_size=(15.0, 15.0, 15.0)
    ).endswith("/s2")


def test_resolve_leaves_scale_array_untouched(multiscale_group):
    array_path = multiscale_group + "/s1"
    assert resolve_scale_path(array_path) == array_path


def test_idi_opens_group_at_default_scale(multiscale_group):
    idi = ImageDataInterface(multiscale_group)
    assert idi.path.endswith("/s0")
    assert tuple(idi.voxel_size) == (4, 4, 4)


def test_idi_opens_group_at_matching_scale(multiscale_group):
    idi = ImageDataInterface(
        multiscale_group, target_voxel_size=(8.0, 8.0, 8.0)
    )
    assert idi.path.endswith("/s1")
    assert tuple(idi.voxel_size) == (8, 8, 8)

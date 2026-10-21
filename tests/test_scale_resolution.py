import zarr
import pytest

from cellmap_analyze.util.image_data_interface import ImageDataInterface


def _make_multiscale_group(path, levels):
    """Create a minimal OME-NGFF multiscale group with isotropic levels.

    ``levels`` is a list of (level_name, voxel_size_float).
    """
    g = zarr.open_group(path, mode="w")
    for name, _ in levels:
        g.create_array(name, shape=(10, 10, 10), dtype="uint8")
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


def test_idi_prefers_finer_over_upsampling(multiscale_group):
    # 15nm sits between the 8nm and 16nm levels. Picking 16nm would require
    # upsampling, so the coarsest level at or finer than the target (8nm) is
    # chosen instead, at its own voxel size.
    idi = ImageDataInterface(
        multiscale_group, target_voxel_size=(15.0, 15.0, 15.0)
    )
    assert idi.path.endswith("/s1")
    assert tuple(idi.voxel_size) == (8, 8, 8)


def test_idi_falls_back_to_finest_when_all_coarser(multiscale_group):
    # Target finer than every level -> upsampling is unavoidable; use s0.
    idi = ImageDataInterface(
        multiscale_group, target_voxel_size=(2.0, 2.0, 2.0)
    )
    assert idi.path.endswith("/s0")


def test_idi_leaves_scale_array_untouched(multiscale_group):
    idi = ImageDataInterface(
        multiscale_group + "/s1", target_voxel_size=(4.0, 4.0, 4.0)
    )
    assert idi.path.endswith("/s1")
    assert tuple(idi.voxel_size) == (8, 8, 8)

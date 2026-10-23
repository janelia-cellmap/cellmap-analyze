"""Tests for zarr v3 format: writing, reading, and backward compatibility with v2."""
import json
import logging
import os

import numpy as np
import pytest

from cellmap_analyze.util.zarr_util import (
    concurrency_context,
    create_multiscale_dataset,
    open_dataset,
)
from cellmap_analyze.util.grid_util import grid_of, make_grid


class TestZarrV3Format:
    def test_new_datasets_are_zarr_v3(self, tmp_path):
        """Newly created datasets should be written in zarr v3 format."""
        output_path = str(tmp_path / "v3test.zarr" / "seg")
        shape = (10, 10, 10)
        ds = create_multiscale_dataset(
            output_path, np.uint8, make_grid(shape, (8, 8, 8), chunks=shape)
        )
        ds.write_index((0, 0, 0), ds.shape, np.ones(shape, dtype=np.uint8))

        # zarr v3 uses zarr.json, zarr v2 uses .zarray
        ds_path = str(tmp_path / "v3test.zarr" / "seg" / "s0")
        assert os.path.exists(os.path.join(ds_path, "zarr.json")), (
            "Expected zarr.json (v3 format) but not found"
        )
        assert not os.path.exists(os.path.join(ds_path, ".zarray")), (
            "Found .zarray (v2 format) — expected v3"
        )

    def test_zarr_v3_roundtrip(self, tmp_path):
        """Write data in v3 format and read it back."""
        output_path = str(tmp_path / "rt.zarr" / "labels")
        shape = (10, 10, 10)
        ds = create_multiscale_dataset(
            output_path, np.uint64, make_grid(shape, (4, 4, 4), chunks=shape)
        )
        test_data = np.zeros(shape, dtype=np.uint64)
        test_data[2:5, 2:5, 2:5] = 1
        test_data[7:9, 7:9, 7:9] = 2
        ds.write_index((0, 0, 0), ds.shape, test_data)

        # Read it back
        vol = open_dataset(output_path + "/s0")
        assert tuple(vol.voxel_size) == (4, 4, 4)
        block = vol.read_index((0, 0, 0), shape)
        assert np.array_equal(block, test_data)

    def test_write_index(self, tmp_path):
        """write_index should work with v3 format."""
        output_path = str(tmp_path / "write.zarr" / "out")
        shape = (5, 5, 5)
        vol = create_multiscale_dataset(
            output_path, np.uint16, make_grid(shape, (8, 8, 8), chunks=shape)
        )
        write_data = np.arange(125, dtype=np.uint16).reshape(shape)
        vol.write_index((0, 0, 0), shape, write_data)

        # Read back
        assert np.array_equal(write_data, vol.read_onto(grid_of(vol)))

    @pytest.mark.parametrize("concurrency_limit", [None, 1, 4])
    def test_writes_take_no_file_lock_or_fsync(self, concurrency_limit):
        """Every writer owns whole chunks, so chunk writes skip tensorstore's per-write
        fcntl lock and its fsync."""
        spec = concurrency_context(concurrency_limit).spec.to_json()
        assert spec["file_io_locking"] == {"mode": "none"}
        assert spec["file_io_sync"] is False


class TestZarrV2BackwardCompat:
    def _create_v2_dataset(self, base_path, ds_name, shape, voxel_size, offset, data):
        """Helper to manually create a zarr v2 dataset on disk."""
        os.makedirs(base_path, exist_ok=True)
        with open(os.path.join(base_path, ".zgroup"), "w") as f:
            json.dump({"zarr_format": 2}, f)

        parts = ds_name.split("/")
        for i in range(len(parts) - 1):
            group_path = os.path.join(base_path, *parts[: i + 1])
            os.makedirs(group_path, exist_ok=True)
            with open(os.path.join(group_path, ".zgroup"), "w") as f:
                json.dump({"zarr_format": 2}, f)

        ds_path = os.path.join(base_path, *parts)
        os.makedirs(ds_path, exist_ok=True)

        zarray = {
            "zarr_format": 2,
            "shape": list(shape),
            "chunks": list(shape),
            "dtype": "|u1",
            "compressor": None,
            "fill_value": 0,
            "order": "C",
            "filters": None,
            "dimension_separator": "/",
        }
        with open(os.path.join(ds_path, ".zarray"), "w") as f:
            json.dump(zarray, f)

        zattrs = {
            "voxel_size": list(voxel_size),
            "offset": list(offset),
        }
        with open(os.path.join(ds_path, ".zattrs"), "w") as f:
            json.dump(zattrs, f)

        # Write raw chunk data (single chunk, uncompressed)
        chunk_path = os.path.join(ds_path, *["0"] * len(shape))
        os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
        data.astype(np.uint8).tofile(chunk_path)

        return ds_path

    def test_read_v2_dataset(self, tmp_path):
        """Should be able to read an existing zarr v2 format dataset."""
        v2_path = str(tmp_path / "legacy.zarr")
        shape = (10, 10, 10)
        vs = (8, 8, 8)
        data = np.ones(shape, dtype=np.uint8) * 42

        self._create_v2_dataset(v2_path, "seg/s0", shape, vs, (0, 0, 0), data)

        # Verify it's actually v2 format
        ds_path = os.path.join(v2_path, "seg", "s0")
        assert os.path.exists(os.path.join(ds_path, ".zarray"))
        assert not os.path.exists(os.path.join(ds_path, "zarr.json"))

        # Read it
        vol = open_dataset(os.path.join(v2_path, "seg/s0"))
        assert tuple(vol.voxel_size) == vs
        assert grid_of(vol).shape == tuple(shape)

        read_data = vol.read_onto(grid_of(vol))
        assert np.all(read_data == 42)

    def test_read_v2_with_ome_metadata(self, tmp_path):
        """Should read OME-Zarr metadata from a zarr v2 parent group."""
        v2_path = str(tmp_path / "ome_v2.zarr")
        shape = (10, 10, 10)
        vs = (8, 8, 8)
        data = np.ones(shape, dtype=np.uint8) * 7

        self._create_v2_dataset(v2_path, "labels/s0", shape, vs, (0, 0, 0), data)

        # Add OME-Zarr metadata on the parent group
        ome_attrs = {
            "multiscales": [{
                "axes": [
                    {"name": "z", "type": "space", "unit": "nanometer"},
                    {"name": "y", "type": "space", "unit": "nanometer"},
                    {"name": "x", "type": "space", "unit": "nanometer"},
                ],
                "datasets": [{
                    "coordinateTransformations": [
                        {"scale": [8.0, 8.0, 8.0], "type": "scale"},
                        {"translation": [0.0, 0.0, 0.0], "type": "translation"},
                    ],
                    "path": "s0",
                }],
                "name": "",
                "version": "0.4",
            }]
        }
        labels_path = os.path.join(v2_path, "labels")
        with open(os.path.join(labels_path, ".zattrs"), "w") as f:
            json.dump(ome_attrs, f)

        vol = open_dataset(os.path.join(v2_path, "labels/s0"))
        assert tuple(vol.voxel_size) == (8.0, 8.0, 8.0)

    def test_read_hybrid_v2_groups_v3_array(self, tmp_path):
        """Should read a v3 array inside v2 group hierarchy with OME metadata."""
        zarr_path = str(tmp_path / "hybrid.zarr")
        os.makedirs(zarr_path)

        # v2 root group
        with open(os.path.join(zarr_path, ".zgroup"), "w") as f:
            json.dump({"zarr_format": 2}, f)

        # v2 intermediate group with OME-Zarr metadata
        seg_path = os.path.join(zarr_path, "seg")
        os.makedirs(seg_path)
        with open(os.path.join(seg_path, ".zgroup"), "w") as f:
            json.dump({"zarr_format": 2}, f)
        ome_attrs = {
            "multiscales": [{
                "axes": [
                    {"name": "z", "type": "space", "unit": "nanometer"},
                    {"name": "y", "type": "space", "unit": "nanometer"},
                    {"name": "x", "type": "space", "unit": "nanometer"},
                ],
                "datasets": [{
                    "coordinateTransformations": [
                        {"scale": [16.0, 16.0, 16.0], "type": "scale"},
                        {"translation": [0.0, 0.0, 0.0], "type": "translation"},
                    ],
                    "path": "s0",
                }],
                "name": "",
                "version": "0.4",
            }]
        }
        with open(os.path.join(seg_path, ".zattrs"), "w") as f:
            json.dump(ome_attrs, f)

        # v3 array (zarr.json, no .zarray) with empty attributes
        import zarr
        arr = zarr.open_array(
            os.path.join(seg_path, "s0"),
            mode="w", shape=(10, 10, 10), chunks=(10, 10, 10), dtype="uint16",
        )
        test_data = np.arange(1000, dtype=np.uint16).reshape(10, 10, 10)
        arr[:] = test_data

        # Verify the hybrid structure
        s0_path = os.path.join(seg_path, "s0")
        assert os.path.exists(os.path.join(s0_path, "zarr.json"))
        assert not os.path.exists(os.path.join(s0_path, ".zarray"))
        assert os.path.exists(os.path.join(seg_path, ".zgroup"))

        # Read via IDI — should get voxel_size from parent OME metadata
        vol = open_dataset(os.path.join(zarr_path, "seg/s0"))
        assert tuple(vol.voxel_size) == (16.0, 16.0, 16.0)
        assert vol.shape == (10, 10, 10)

        # Verify data round-trip. Read the dataset's own ROI: the OME
        # translation is the voxel CENTER, so the corner is translation - vs/2,
        # not the origin.
        read_data = vol.read_onto(grid_of(vol))
        assert np.array_equal(read_data, test_data)

    def test_v3_output_has_ome_ngff_metadata(self, tmp_path):
        """Output from create_multiscale_dataset should have proper OME-NGFF v3."""
        import zarr
        output_path = str(tmp_path / "ngff.zarr" / "mito")
        shape = (10, 10, 10)
        create_multiscale_dataset(
            output_path, np.uint8, make_grid(shape, (16, 16, 16), chunks=shape)
        )

        # Parent group should have OME-NGFF 0.5 multiscales in zarr.json
        group = zarr.open_group(output_path, mode="r")
        ome = dict(group.attrs)["ome"]
        assert ome["version"] == "0.5"
        ms = ome["multiscales"][0]
        assert len(ms["axes"]) == 3
        assert ms["axes"][0]["unit"] == "nanometer"

        scale = ms["datasets"][0]["coordinateTransformations"][0]
        assert scale["type"] == "scale"
        assert scale["scale"] == [16, 16, 16]

        # s0 array should be zarr v3 format
        s0_path = os.path.join(str(tmp_path / "ngff.zarr" / "mito" / "s0"))
        assert os.path.exists(os.path.join(s0_path, "zarr.json"))
        assert not os.path.exists(os.path.join(s0_path, ".zarray"))

    def test_outputs_match_container_format(self, tmp_path):
        """Outputs should be created in the container's zarr format.

        A v2 container (root .zgroup) should get v2 arrays (.zarray),
        not v3 arrays (zarr.json). This ensures tools like Neuroglancer
        that inherit the format from the root group can read the arrays.
        """
        # Create a v2 container
        v2_path = str(tmp_path / "v2container.zarr")
        os.makedirs(v2_path)
        with open(os.path.join(v2_path, ".zgroup"), "w") as f:
            json.dump({"zarr_format": 2}, f)
        with open(os.path.join(v2_path, ".zattrs"), "w") as f:
            json.dump({}, f)

        shape = (10, 10, 10)
        grid = make_grid(shape, (8, 8, 8), chunks=shape)
        create_multiscale_dataset(f"{v2_path}/seg", np.uint8, grid)

        # Array should be v2 format
        s0_path = os.path.join(v2_path, "seg", "s0")
        assert os.path.exists(os.path.join(s0_path, ".zarray")), (
            "Expected .zarray (v2) in v2 container but not found"
        )
        assert not os.path.exists(os.path.join(s0_path, "zarr.json")), (
            "Found zarr.json (v3) in v2 container — format mismatch"
        )

        # Create a v3 container
        v3_path = str(tmp_path / "v3container.zarr")
        import zarr
        zarr.open_group(v3_path, mode="w", zarr_format=3)

        create_multiscale_dataset(f"{v3_path}/seg", np.uint8, grid)

        # Array should be v3 format
        s0_path = os.path.join(v3_path, "seg", "s0")
        assert os.path.exists(os.path.join(s0_path, "zarr.json")), (
            "Expected zarr.json (v3) in v3 container but not found"
        )
        assert not os.path.exists(os.path.join(s0_path, ".zarray")), (
            "Found .zarray (v2) in v3 container — format mismatch"
        )


class TestMultiscaleLevelSelection:
    """Each scale level (s0, s1, ...) carries its own scale + translation in the
    OME multiscales metadata. Opening a non-s0 level of an external dataset
    (no per-array attrs) must read THAT level's transform, not s0's."""

    def _make_external_multiscale(self, root):
        """Two-level OME multiscale on the parent group, arrays carry no
        per-array voxel_size/offset attrs (pure OME, like external EM data)."""
        import zarr

        seg = os.path.join(root, "seg")
        os.makedirs(seg)
        json.dump({"zarr_format": 2}, open(os.path.join(root, ".zgroup"), "w"))
        json.dump({"zarr_format": 2}, open(os.path.join(seg, ".zgroup"), "w"))
        ome = {
            "multiscales": [{
                "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
                "datasets": [
                    {"path": "s0", "coordinateTransformations": [
                        {"scale": [4.0, 4.0, 4.0], "type": "scale"},
                        {"translation": [0.0, 0.0, 0.0], "type": "translation"}]},
                    {"path": "s1", "coordinateTransformations": [
                        {"scale": [8.0, 8.0, 8.0], "type": "scale"},
                        {"translation": [2.0, 2.0, 2.0], "type": "translation"}]},
                ],
                "name": "", "version": "0.4",
            }]
        }
        json.dump(ome, open(os.path.join(seg, ".zattrs"), "w"))
        for name, shp in [("s0", (20, 20, 20)), ("s1", (10, 10, 10))]:
            a = zarr.open_array(
                os.path.join(seg, name), mode="w", shape=shp, chunks=shp,
                dtype="uint8", zarr_format=2,
            )
            a[:] = 1
        return seg

    def test_reads_per_level_scale_and_translation(self, tmp_path):
        seg = self._make_external_multiscale(str(tmp_path / "ext.zarr"))

        vol0 = open_dataset(f"{seg}/s0")
        assert tuple(vol0.voxel_size) == (4, 4, 4)
        # translation 0 (center) -> corner = 0 - 4/2 = -2
        assert tuple(vol0.corner) == (-2, -2, -2)

        vol1 = open_dataset(f"{seg}/s1")
        # must pick up s1's scale, not s0's
        assert tuple(vol1.voxel_size) == (8, 8, 8)
        # translation 2 (center) -> corner = 2 - 8/2 = -2
        assert tuple(vol1.corner) == (-2, -2, -2)

        # Both levels share the same corner (-V_base/2) -- the OME pyramid
        # invariant that keeps scales aligned.
        assert tuple(vol0.corner) == tuple(vol1.corner)


class TestMetadataPrecedence:
    """OME multiscales is the canonical source for voxel_size / translation;
    funlib-style per-array attrs are accepted as a legacy fallback. When both
    are present and they DISAGREE, reading must raise loudly -- silent
    precedence in this case is exactly how stale per-array attrs can override
    a corrected group-level OME and silently misplace downstream operations.
    """

    def _make_group_with_array(
        self, root, ome=None, per_array=None, vs=(8, 8, 8), shape=(4, 4, 4)
    ):
        """Build a v2 zarr group at ``root`` with a single child array ``s0``.

        ``ome``: dict to write as the group's .zattrs (typically multiscales).
        ``per_array``: dict to write as the s0 array's .zattrs.
        """
        import zarr

        os.makedirs(root)
        json.dump({"zarr_format": 2}, open(os.path.join(root, ".zgroup"), "w"))
        if ome is not None:
            json.dump(ome, open(os.path.join(root, ".zattrs"), "w"))
        a = zarr.open_array(
            os.path.join(root, "s0"),
            mode="w",
            shape=shape,
            chunks=shape,
            dtype="uint8",
            zarr_format=2,
        )
        a[:] = 1
        if per_array is not None:
            # zarr v2 writes .zattrs automatically when you set attrs; do it
            # explicitly so we control exactly what lands on disk.
            json.dump(
                per_array, open(os.path.join(root, "s0", ".zattrs"), "w")
            )
        return os.path.join(root, "s0")

    def _ome(self, vs=(8.0, 8.0, 8.0), trans=(0.0, 0.0, 0.0)):
        return {
            "multiscales": [
                {
                    "axes": [
                        {"name": a, "type": "space", "unit": "nanometer"}
                        for a in "zyx"
                    ],
                    "datasets": [
                        {
                            "path": "s0",
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
        }

    def test_ome_only_returns_ome_values(self, tmp_path):
        """No per-array attrs at all -> OME is read."""
        s0 = self._make_group_with_array(
            str(tmp_path / "ome_only.zarr"),
            ome=self._ome(vs=(8.0, 8.0, 8.0), trans=(4.0, 4.0, 4.0)),
            per_array=None,
        )
        vol = open_dataset(s0)
        assert tuple(vol.voxel_size) == (8, 8, 8)
        # corner = translation - vs/2 = 4 - 4 = 0
        assert tuple(vol.corner) == (0, 0, 0)

    def test_per_array_only_returns_per_array_values(self, tmp_path):
        """No OME multiscales -> falls back to per-array attrs, read as funlib writes
        them: the offset is the corner of voxel 0."""
        s0 = self._make_group_with_array(
            str(tmp_path / "attr_only.zarr"),
            ome=None,
            per_array={"voxel_size": [8.0, 8.0, 8.0], "offset": [4.0, 4.0, 4.0]},
        )
        vol = open_dataset(s0)
        assert tuple(vol.voxel_size) == (8, 8, 8)
        assert tuple(vol.corner) == (4, 4, 4)

    def test_both_consistent_succeeds(self, tmp_path):
        """OME and per-array attrs both present and EQUAL -> no error, value
        is returned (this is what cellmap-written datasets look like)."""
        s0 = self._make_group_with_array(
            str(tmp_path / "consistent.zarr"),
            ome=self._ome(vs=(8.0, 8.0, 8.0), trans=(4.0, 4.0, 4.0)),
            per_array={"voxel_size": [8.0, 8.0, 8.0], "offset": [4.0, 4.0, 4.0]},
        )
        vol = open_dataset(s0)
        assert tuple(vol.voxel_size) == (8, 8, 8)
        assert tuple(vol.corner) == (0, 0, 0)

    def test_voxel_size_conflict_raises(self, tmp_path):
        """OME scale and per-array voxel_size disagree -> ValueError that
        names BOTH values so the user can reconcile."""
        s0 = self._make_group_with_array(
            str(tmp_path / "vs_conflict.zarr"),
            ome=self._ome(vs=(16.0, 16.0, 16.0), trans=(6.0, 6.0, 6.0)),
            per_array={"voxel_size": [8.0, 8.0, 8.0], "offset": [6.0, 6.0, 6.0]},
        )
        with pytest.raises(ValueError) as exc:
            open_dataset(s0)
        msg = str(exc.value)
        assert "voxel_size" in msg
        assert "16" in msg and "8" in msg
        assert "OME" in msg

    def test_translation_conflict_warns(self, tmp_path, caplog):
        """OME translation and per-array offset disagree -> a warning naming both, and
        the OME translation places the array."""
        s0 = self._make_group_with_array(
            str(tmp_path / "tr_conflict.zarr"),
            ome=self._ome(vs=(16.0, 16.0, 16.0), trans=(6.0, 6.0, 6.0)),
            per_array={"voxel_size": [16.0, 16.0, 16.0], "offset": [0.0, 0.0, 0.0]},
        )
        with caplog.at_level(logging.WARNING, logger="worldarray"):
            vol = open_dataset(s0)
        assert "offset [0.0, 0.0, 0.0]" in caplog.text
        assert "OME translation [6.0, 6.0, 6.0]" in caplog.text
        assert tuple(vol.corner) == (-2, -2, -2)  # 6 - 16 / 2

    def test_conflict_message_includes_path(self, tmp_path, caplog):
        """The warning names the dataset path for diagnosis."""
        target = str(tmp_path / "named.zarr")
        s0 = self._make_group_with_array(
            target,
            ome=self._ome(vs=(16.0, 16.0, 16.0), trans=(6.0, 6.0, 6.0)),
            per_array={"voxel_size": [16.0, 16.0, 16.0], "offset": [0.0, 0.0, 0.0]},
        )
        with caplog.at_level(logging.WARNING, logger="worldarray"):
            open_dataset(s0)
        assert "named.zarr" in caplog.text

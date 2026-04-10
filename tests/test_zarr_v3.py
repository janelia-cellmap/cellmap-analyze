"""Tests for zarr v3 format: writing, reading, and backward compatibility with v2."""
import json
import os

import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi

from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.zarr_util import (
    create_multiscale_dataset,
    create_multiscale_dataset_idi,
)


class TestZarrV3Format:
    def test_new_datasets_are_zarr_v3(self, tmp_path):
        """Newly created datasets should be written in zarr v3 format."""
        output_path = str(tmp_path / "v3test.zarr" / "seg")
        vs = Coordinate(8, 8, 8)
        shape = (10, 10, 10)
        roi = Roi(Coordinate(0, 0, 0), Coordinate(shape) * vs)

        ds = create_multiscale_dataset(
            output_path,
            dtype=np.uint8,
            voxel_size=vs,
            total_roi=roi,
            write_size=roi.shape,
        )
        ds.data[:] = np.ones(shape, dtype=np.uint8)

        # zarr v3 uses zarr.json, zarr v2 uses .zarray
        ds_path = str(tmp_path / "v3test.zarr" / "seg" / "s0")
        assert os.path.exists(os.path.join(ds_path, "zarr.json")), (
            "Expected zarr.json (v3 format) but not found"
        )
        assert not os.path.exists(os.path.join(ds_path, ".zarray")), (
            "Found .zarray (v2 format) — expected v3"
        )

    def test_zarr_v3_roundtrip_via_idi(self, tmp_path):
        """Write data in v3 format and read it back via ImageDataInterface."""
        output_path = str(tmp_path / "rt.zarr" / "labels")
        vs = Coordinate(4, 4, 4)
        shape = (10, 10, 10)
        roi = Roi(Coordinate(0, 0, 0), Coordinate(shape) * vs)

        ds = create_multiscale_dataset(
            output_path,
            dtype=np.uint64,
            voxel_size=vs,
            total_roi=roi,
            write_size=roi.shape,
        )
        test_data = np.zeros(shape, dtype=np.uint64)
        test_data[2:5, 2:5, 2:5] = 1
        test_data[7:9, 7:9, 7:9] = 2
        ds.data[:] = test_data

        # Read back via IDI
        idi = ImageDataInterface(output_path + "/s0")
        assert tuple(idi.voxel_size) == (4, 4, 4)
        block = idi.to_ndarray_ds(roi)
        assert np.array_equal(block, test_data)

    def test_write_via_roi_setitem(self, tmp_path):
        """The idi.ds[roi] = value pattern should work with v3 format."""
        output_path = str(tmp_path / "write.zarr" / "out")
        vs = Coordinate(8, 8, 8)
        shape = (5, 5, 5)
        roi = Roi(Coordinate(0, 0, 0), Coordinate(shape) * vs)

        idi = create_multiscale_dataset_idi(
            output_path, dtype=np.uint16, voxel_size=vs,
            total_roi=roi, write_size=roi.shape,
        )
        write_data = np.arange(125, dtype=np.uint16).reshape(shape)
        idi.ds[roi] = write_data

        # Read back
        readback = np.array(idi.ds.data[:])
        assert np.array_equal(write_data, readback)


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

        # Read via ImageDataInterface
        idi = ImageDataInterface(os.path.join(v2_path, "seg/s0"))
        assert tuple(idi.voxel_size) == vs
        assert tuple(idi.roi.shape) == tuple(s * v for s, v in zip(shape, vs))

        roi = Roi(Coordinate(0, 0, 0), Coordinate(shape) * Coordinate(vs))
        read_data = idi.to_ndarray_ds(roi)
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

        idi = ImageDataInterface(os.path.join(v2_path, "labels/s0"))
        assert tuple(idi.original_voxel_size) == (8.0, 8.0, 8.0)

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
        idi = ImageDataInterface(os.path.join(zarr_path, "seg/s0"))
        assert tuple(idi.original_voxel_size) == (16.0, 16.0, 16.0)
        assert idi.ds.data.shape == (10, 10, 10)

        # Verify data round-trip
        from funlib.geometry import Roi, Coordinate
        roi = Roi(Coordinate(0, 0, 0), Coordinate(10, 10, 10) * idi.voxel_size)
        read_data = idi.to_ndarray_ds(roi)
        assert np.array_equal(read_data, test_data)

    def test_v3_output_has_ome_ngff_metadata(self, tmp_path):
        """Output from create_multiscale_dataset should have proper OME-NGFF v3."""
        import zarr
        output_path = str(tmp_path / "ngff.zarr" / "mito")
        vs = Coordinate(16, 16, 16)
        shape = (10, 10, 10)
        roi = Roi(Coordinate(0, 0, 0), Coordinate(shape) * vs)

        create_multiscale_dataset(
            output_path, dtype=np.uint8, voxel_size=vs,
            total_roi=roi, write_size=roi.shape,
        )

        # Parent group should have OME-NGFF multiscales in zarr.json
        group = zarr.open_group(output_path, mode="r")
        attrs = dict(group.attrs)
        assert "multiscales" in attrs
        ms = attrs["multiscales"][0]
        assert ms["version"] == "0.4"
        assert len(ms["axes"]) == 3
        assert ms["axes"][0]["unit"] == "nanometer"

        scale = ms["datasets"][0]["coordinateTransformations"][0]
        assert scale["type"] == "scale"
        assert scale["scale"] == [16, 16, 16]

        # s0 array should be zarr v3 format
        s0_path = os.path.join(str(tmp_path / "ngff.zarr" / "mito" / "s0"))
        assert os.path.exists(os.path.join(s0_path, "zarr.json"))
        assert not os.path.exists(os.path.join(s0_path, ".zarray"))

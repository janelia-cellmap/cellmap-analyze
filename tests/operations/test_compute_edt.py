import edt as edt_module
import numpy as np

from cellmap_analyze.process.compute_edt import ComputeEDT
from cellmap_analyze.util.grid_util import make_grid
from cellmap_analyze.util.zarr_util import create_multiscale_dataset
from tests.test_utils import read_all


def _write_segmentation(path, seg, voxel_size=(1.0, 1.0, 1.0)):
    ds = create_multiscale_dataset(
        path,
        dtype=seg.dtype,
        grid=make_grid(seg.shape, voxel_size, chunks=(10, 10, 10)),
    )
    ds.write_index((0, 0, 0), ds.shape, seg)
    return f"{path}/s0"


def _multi_instance_segmentation():
    # Two touching-but-distinct instances (different ids, no background gap
    # between them) plus a separate instance elsewhere -- exercises the
    # multi-label-aware behavior (a same-id-vs-different-id boundary, not
    # just foreground-vs-background).
    seg = np.zeros((30, 30, 30), dtype=np.uint32)
    seg[2:15, 2:15, 2:15] = 1
    seg[15:25, 2:15, 2:15] = 2
    seg[2:10, 20:28, 20:28] = 3
    return seg


def test_compute_edt_matches_whole_array_multi_label(tmp_path):
    seg = _multi_instance_segmentation()
    seg_path = _write_segmentation(str(tmp_path / "segmentation.zarr"), seg)
    output_path = str(tmp_path / "edt_output.zarr")

    cedt = ComputeEDT(
        segmentation_path=seg_path,
        output_path=output_path,
        padding_nm=8,
        block_shape_voxels=(12, 12, 12),
        num_workers=1,
    )
    cedt.calculate_edt()

    blockwise = read_all(f"{output_path}/s0")
    whole_array = edt_module.edt(seg, anisotropy=(1.0, 1.0, 1.0))

    # Padding (8) comfortably exceeds the block-tiling granularity here, so
    # blockwise should reproduce the exact whole-array multi-label EDT.
    assert np.allclose(blockwise, whole_array, atol=1e-4)

    # Sanity: the boundary between instances 1 and 2 (touching, no background
    # gap) should show up as a *thin* region, not be invisible the way a
    # foreground-vs-background-only EDT would render it.
    assert blockwise[14, 8, 8] <= 1.0001
    assert blockwise[15, 8, 8] <= 1.0001

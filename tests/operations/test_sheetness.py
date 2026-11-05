import numpy as np
import pandas as pd
import pytest

from cellmap_analyze.analyze.sheetness import (
    Sheetness,
    from_level,
    hessian_sheetness,
    scale_series,
    to_level,
)
from cellmap_analyze.cythonizing.geodesic import nearest_source_values
from cellmap_analyze.cythonizing.topological_thinning import thin
from cellmap_analyze.process.topological_thinning import TopologicalThinning
from cellmap_analyze.util.grid_util import make_grid
from cellmap_analyze.util.zarr_util import create_multiscale_dataset, open_dataset

VOXEL_SIZE = (8.0, 8.0, 8.0)
SHAPE = (64, 100, 100)
SCALE_STEPS = 8


def _sheet_and_tubes():
    """A sheet 5 voxels thick (ID 1) and three tubes 5 voxels wide (ID 2)."""
    zz, yy, xx = np.indices(SHAPE)
    labels = np.zeros(SHAPE, np.uint64)
    labels[(np.abs(zz - 20) <= 2) & (yy > 10) & (yy < 90) & (xx > 10) & (xx < 90)] = 1
    for y0 in (30, 50, 70):
        labels[((zz - 45) ** 2 + (yy - y0) ** 2 <= 6.5) & (xx > 10) & (xx < 90)] = 2
    return labels


def _medial_surface(labels):
    padded = np.pad(labels, 1)
    thin(padded, (-1, -1, -1), True, 0)
    return padded[1:-1, 1:-1, 1:-1]


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sheetness")
    labels = _sheet_and_tubes()
    grid = make_grid(SHAPE, VOXEL_SIZE, chunks=(16, 20, 20))
    create_multiscale_dataset(f"{tmp}/in.zarr/er", dtype=np.uint64, grid=grid).write_index(
        (0, 0, 0), SHAPE, labels
    )
    TopologicalThinning(
        f"{tmp}/in.zarr/er",
        f"{tmp}/in.zarr/er_medial_surface",
        medial_surface=True,
        block_shape_voxels=(64, 60, 60),
        num_workers=1,
    ).thin()
    return tmp, labels


def test_sheets_score_high_and_tubes_low():
    labels = _sheet_and_tubes()
    medial = _medial_surface(labels)
    increments, normalisation = scale_series(VOXEL_SIZE, SCALE_STEPS)
    # away from rims and tube ends
    core = np.zeros(SHAPE, bool)
    core[:, 25:75, 25:75] = True
    for id_value, low, high in ((1, 0.7, 0.9), (2, 0.05, 0.25)):
        points = np.argwhere((medial == id_value) & core)
        sheetness = hessian_sheetness(labels == id_value, points, increments, normalisation)
        assert low < np.median(sheetness) < high


def test_nearest_source_values_stay_in_their_object():
    labels = np.zeros((3, 3, 12), np.uint8)
    labels[1, 1, :6] = 1
    labels[1, 1, 6:] = 2
    sources = np.array([[1, 1, 0], [1, 1, 4], [1, 1, 11]])
    distance, values = nearest_source_values(labels, sources, np.array([0.2, 0.6, 1.0]), VOXEL_SIZE)
    line = values[1, 1]
    # voxel 2 is as near to both sources of object 1, so takes their mean
    assert np.allclose(line[:6], [0.2, 0.2, 0.4, 0.6, 0.6, 0.6])
    assert np.allclose(line[6:], 1.0)
    assert np.allclose(distance[1, 1, :3], [0, 8, 16])
    assert np.isnan(values[0, 0, 0]) and np.isinf(distance[0, 0, 0])


def test_levels_round_trip():
    sheetness = np.linspace(0, 1, 255)
    assert np.allclose(from_level(to_level(sheetness)), sheetness, atol=0.5 / 254)
    assert to_level(np.array([0.0]))[0] == 1 and to_level(np.array([1.0]))[0] == 255


@pytest.mark.parametrize("block_shape", [(32, 40, 40), (64, 100, 60)])
def test_sheetness_per_object(datasets, tmp_path, block_shape):
    tmp, labels = datasets
    sheetness = Sheetness(
        f"{tmp}/in.zarr/er",
        f"{tmp}/in.zarr/er_medial_surface",
        f"{tmp_path}/out.zarr/er_sheetness",
        f"{tmp_path}/csv",
        scale_steps=SCALE_STEPS,
        block_shape_voxels=block_shape,
        num_workers=1,
    )
    sheetness.calculate()
    voxel_volume = np.prod(VOXEL_SIZE)

    summary = pd.read_csv(f"{tmp_path}/csv/er_sheetness.csv", index_col=0)
    assert list(summary.index) == [1, 2]
    for id_value in (1, 2):
        assert summary.loc[id_value, "Volume (nm^3)"] == (labels == id_value).sum() * voxel_volume
    assert np.all(summary["Unreached Volume (nm^3)"] == 0)
    assert summary.loc[1, "Mean Sheetness"] > 0.6
    assert summary.loc[2, "Mean Sheetness"] < 0.25
    # 2 x the distance to the nearest outside voxel centre: 6 voxels for a
    # sheet 5 thick
    assert summary.loc[1, "Mean Thickness (nm)"] == pytest.approx(48, abs=2)

    level = open_dataset(f"{tmp_path}/out.zarr/er_sheetness/s0").read_index((0, 0, 0), SHAPE)
    assert np.all((level > 0) == (labels > 0))

    histograms = pd.read_csv(f"{tmp_path}/csv/er_sheetness_histograms.csv")
    by_object = histograms.groupby("Object ID")[["Volume (nm^3)", "Surface Area (nm^2)"]].sum()
    assert np.allclose(by_object["Volume (nm^3)"], summary["Volume (nm^3)"])
    assert np.allclose(by_object["Surface Area (nm^2)"], summary["Surface Area (nm^2)"])

    vs_thickness = pd.read_csv(f"{tmp_path}/csv/er_sheetness_vs_thickness.csv")
    medial = open_dataset(f"{tmp}/in.zarr/er_medial_surface/s0").read_index((0, 0, 0), SHAPE)
    counts = vs_thickness.groupby("Object ID")["Medial Surface Voxels"].sum()
    assert counts[1] == (medial == 1).sum() and counts[2] == (medial == 2).sum()

"""CustomSkeleton.simplify keeps every cycle of the skeleton it simplifies."""
import networkx as nx
import numpy as np
import pytest
from pybind11_rdp import rdp

from cellmap_analyze.util.skeleton_util import CustomSkeleton

TOLERANCE_NM = 5


def _ring(width, length=200.0, n=40):
    """n vertices around a closed loop `length` long and `width` wide, from x = length / 2."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(length / 2 * np.cos(a), width / 2 * np.sin(a), 0.0) for a in t]


def ring(width):
    """An isolated ring: no branch points or endpoints."""
    return CustomSkeleton(_ring(width), [(i, (i + 1) % 40) for i in range(40)])


def lollipop(width):
    """A loop closed at the branch point where a stem leaves it."""
    stem = [(100.0 + 10 * i, 0.0, 0.0) for i in range(1, 6)]
    edges = [(i, (i + 1) % 40) for i in range(40)] + [(0, 40)]
    edges += [(40 + i, 41 + i) for i in range(4)]
    return CustomSkeleton(_ring(width) + stem, edges)


def theta(width):
    """A ring with a stem at each end: two parallel polylines between the two branch points."""
    stems = [(100.0 + 10 * i, 0.0, 0.0) for i in range(1, 6)]
    stems += [(-100.0 - 10 * i, 0.0, 0.0) for i in range(1, 6)]
    edges = [(i, (i + 1) % 40) for i in range(40)] + [(0, 40), (20, 45)]
    edges += [(40 + i, 41 + i) for i in range(4)] + [(45 + i, 46 + i) for i in range(4)]
    return CustomSkeleton(_ring(width) + stems, edges)


def _cycles(skeleton):
    g = nx.Graph()
    g.add_nodes_from(range(len(skeleton.vertices)))
    g.add_edges_from(tuple(e) for e in skeleton.edges if e[0] != e[1])
    return g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g)


@pytest.mark.parametrize("make", [ring, lollipop, theta])
@pytest.mark.parametrize("width", [8.0, 1.0])
def test_loops_narrower_than_the_tolerance_stay_loops(make, width):
    # RDP alone reduces each of these loops to a line out and back (or, for the
    # theta, both arcs to the same line), losing the cycle.
    skeleton = make(width)
    simplified = skeleton.simplify(TOLERANCE_NM)
    assert _cycles(simplified) == _cycles(skeleton) == 1


@pytest.mark.parametrize("make", [ring, lollipop, theta])
def test_loops_wider_than_the_tolerance_are_plain_rdp(make):
    # Nothing is restored when RDP already keeps the cycles.
    skeleton = make(60.0)
    simplified = skeleton.simplify(TOLERANCE_NM)
    polylines = skeleton.get_polylines_positions_from_graph(skeleton.skeleton_to_graph())
    rdp_vertices = {
        tuple(np.round(v, 6)) for p in polylines for v in rdp(np.asarray(p, float), epsilon=TOLERANCE_NM)
    }
    assert {tuple(np.round(v, 6)) for v in simplified.vertices} == rdp_vertices
    assert _cycles(simplified) == 1

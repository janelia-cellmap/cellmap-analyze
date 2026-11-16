from dataclasses import dataclass
import struct
import numpy as np
import os
import logging
import time
import itertools
from neuroglancer.skeleton import (
    Skeleton as NeuroglancerSkeleton,
    VertexAttributeInfo,
)
import fastremap
import networkx as nx
from pybind11_rdp import rdp

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Offsets to the 26 neighbours of a voxel, one of each +/- pair, so every
# adjacency is visited once.
_HALF_OFFSETS = [
    np.array(o, dtype=np.int64)
    for o in itertools.product((-1, 0, 1), repeat=3)
    if o > (0, 0, 0)
]


def skimage_to_custom_skeleton_fast(binary_skeleton, spacing=(1, 1, 1), simplify_triangles=True):
    """Turn a thinned binary skeleton into a CustomSkeleton (vertices = skeleton
    voxels at ``index * spacing``, edges between 26-adjacent voxels).

    The edges are built so that a cycle in the graph means a tunnel in the
    skeleton, and nothing else:

    1. two voxels are not joined when a third skeleton voxel lies in the box
       they span (it already joins them with shorter steps);
    2. a triangle that remains loses its longest edge;
    3. a cycle inside a 3x3x3 block around a junction is reduced to a
       spanning forest (such a cycle encloses at most one voxel).

    Without this, a hub voxel and its face neighbours form a ring of diagonal
    adjacencies that is a cycle of the graph but no tunnel of the skeleton;
    each such ring adds two junctions and up to three branches to the count,
    and pruning cannot remove it.

    Args:
        binary_skeleton: Binary 3D array (output of skimage.morphology.skeletonize).
        spacing: Voxel spacing in each dimension.
        simplify_triangles: Apply steps 1-3. ``False`` keeps every 26-adjacency.
    """
    start_time = time.time()
    coords = np.argwhere(binary_skeleton)
    logger.info(f"skimage_to_custom_skeleton_fast: {len(coords)} skeleton voxels")

    if len(coords) == 0:
        logger.info("No coordinates found, returning empty skeleton")
        return CustomSkeleton()

    positions = coords * np.asarray(spacing, dtype=float)
    vertices = [tuple(v) for v in positions]

    if simplify_triangles:
        edges = skeleton_voxel_edges(coords)
        edges = simplify_triangle_cliques(edges, len(coords), vertices=positions)
        edges = break_block_cycles(edges, coords, positions)
    else:
        edges = all_26_adjacencies(coords)
    edges = [(int(a), int(b)) for a, b in edges]
    logger.info(
        f"Built {len(edges)} edges in {time.time() - start_time:.4f}s "
        f"(simplify_triangles={simplify_triangles})"
    )

    skeleton = CustomSkeleton(vertices=vertices, edges=edges)
    logger.info(f"Total skimage_to_custom_skeleton_fast time: {time.time() - start_time:.4f}s")
    return skeleton


def _voxel_index(coords):
    """-> (lookup, shifted): ``lookup(points)`` gives the index into ``coords`` of
    each point of the shifted grid, or -1 where there is no skeleton voxel.
    ``shifted`` is ``coords`` moved so that every 26-neighbour is inside the grid."""
    coords = np.asarray(coords, dtype=np.int64)
    n = len(coords)
    shifted = coords - coords.min(axis=0) + 1
    shape = tuple(int(s) for s in shifted.max(axis=0) + 2)
    linear = np.ravel_multi_index(shifted.T, shape)
    order = np.argsort(linear)
    linear_sorted = linear[order]

    def lookup(points):
        l = np.ravel_multi_index(np.asarray(points).T, shape)
        pos = np.minimum(np.searchsorted(linear_sorted, l), n - 1)
        return np.where(linear_sorted[pos] == l, order[pos], -1)

    return lookup, shifted


def all_26_adjacencies(coords):
    """Every pair of 26-adjacent skeleton voxels, as (i, j) index pairs with i < j."""
    coords = np.asarray(coords, dtype=np.int64)
    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    lookup, shifted = _voxel_index(coords)
    own = np.arange(len(coords))
    edges = []
    for offset in _HALF_OFFSETS:
        neighbour = lookup(shifted + offset)
        joined = neighbour >= 0
        edges.append(np.column_stack([own[joined], neighbour[joined]]))
    return np.concatenate(edges)


def skeleton_voxel_edges(coords):
    """Edges of the voxel graph of a thinned skeleton, as (i, j) index pairs into
    ``coords``.

    Two skeleton voxels are joined when they are 26-adjacent and no third
    skeleton voxel lies in the box the pair spans. A voxel in that box is
    face- or edge-adjacent to both, so it already joins them with shorter
    steps, and the direct diagonal would only add a cycle that encloses
    nothing. Face-adjacent pairs span no box and are always joined.
    """
    coords = np.asarray(coords, dtype=np.int64)
    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    lookup, shifted = _voxel_index(coords)
    own = np.arange(len(coords))
    edges = []
    for offset in _HALF_OFFSETS:
        neighbour = lookup(shifted + offset)
        joined = neighbour >= 0
        axes = np.flatnonzero(offset)
        if len(axes) > 1:
            bridged = np.zeros(len(coords), dtype=bool)
            # the corners of the box spanned by the pair, other than the pair itself
            for bits in itertools.product((0, 1), repeat=len(axes)):
                if not any(bits) or all(bits):
                    continue
                corner = np.zeros(3, dtype=np.int64)
                for axis, bit in zip(axes, bits):
                    if bit:
                        corner[axis] = offset[axis]
                bridged |= lookup(shifted + corner) >= 0
            joined &= ~bridged
        edges.append(np.column_stack([own[joined], neighbour[joined]]))
    return np.concatenate(edges)


def _edge_length(vertices, a, b):
    if vertices is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(vertices[a], float) - np.asarray(vertices[b], float)))


def simplify_triangle_cliques(edges, num_vertices, vertices=None):
    """Break every triangle (three mutually joined vertices) by dropping its
    longest edge; a triangle of skeleton voxels encloses nothing, so it is
    never a tunnel. Ties, and the case without ``vertices``, are decided by
    vertex index so the result is deterministic. Returns the edges as a list
    of sorted (i, j) tuples.
    """
    g = nx.Graph()
    g.add_nodes_from(range(num_vertices))
    g.add_edges_from((int(a), int(b)) for a, b in edges)

    triangles = [clique for clique in nx.enumerate_all_cliques(g) if len(clique) == 3]
    if triangles:
        logger.info(f"Breaking {len(triangles)} triangles")
    for a, b, c in triangles:
        candidates = [(a, b), (b, c), (a, c)]
        if all(g.has_edge(*e) for e in candidates):
            g.remove_edge(
                *max(candidates, key=lambda e: (_edge_length(vertices, *e), tuple(sorted(e))))
            )
    return [tuple(sorted(e)) for e in g.edges]


def break_block_cycles(edges, coords, vertices, block=3):
    """Inside a ``block``^3 group of voxels a cycle encloses at most one voxel,
    so it is either no tunnel at all or a one-voxel pinhole; neither is a
    feature of the shape. For every such block around a junction (a vertex
    with three or more edges), keep only a shortest spanning forest of the
    edges between the block's voxels. A cycle that touches no junction is a
    component of its own and is left alone. Returns a list of sorted (i, j)
    tuples.
    """
    coords = np.asarray(coords, dtype=np.int64)
    g = nx.Graph()
    g.add_nodes_from(range(len(coords)))
    for a, b in edges:
        g.add_edge(int(a), int(b), weight=_edge_length(vertices, a, b))
    coord_to_idx = {tuple(c): i for i, c in enumerate(coords)}
    cells = [np.array(o, dtype=np.int64) for o in itertools.product(range(block), repeat=3)]
    removed = 0
    for junction in [v for v in g if g.degree[v] >= 3]:
        for offset in cells:
            corner = coords[junction] - offset
            nodes = [
                coord_to_idx[q]
                for q in (tuple(corner + cell) for cell in cells)
                if q in coord_to_idx
            ]
            if len(nodes) < 3:
                continue
            sub = g.subgraph(nodes)
            if sub.number_of_edges() <= len(nodes) - nx.number_connected_components(sub):
                continue  # already a forest
            keep = {frozenset(e) for e in nx.minimum_spanning_tree(sub, weight="weight").edges}
            drop = [e for e in sub.edges if frozenset(e) not in keep]
            g.remove_edges_from(drop)
            removed += len(drop)
    if removed:
        logger.info(f"Removed {removed} edges closing cycles inside {block}x{block}x{block} blocks")
    return [tuple(sorted(e)) for e in g.edges]


@dataclass
class Source:
    vertex_attributes = []


class CustomSkeleton:
    def __init__(self, vertices=[], edges=[], radii=None, polylines=[]):
        start_time = time.time()
        logger.info(
            f"Initializing CustomSkeleton with {len(vertices)} vertices, {len(edges)} edges"
        )

        self.vertices = []
        self.edges = []
        self.radii = []
        self.polylines = []

        add_vertices_start = time.time()
        self.add_vertices(vertices, radii=radii)
        logger.info(f"Added vertices in {time.time() - add_vertices_start:.4f}s")

        add_edges_start = time.time()
        self.add_edges(edges)
        logger.info(f"Added edges in {time.time() - add_edges_start:.4f}s")

        if not polylines:
            graph_start = time.time()
            g = self.skeleton_to_graph()
            logger.info(f"Created graph in {time.time() - graph_start:.4f}s")

            polylines_start = time.time()
            polylines = self.get_polylines_positions_from_graph(g)
            logger.info(
                f"Extracted {len(polylines)} polylines in {time.time() - polylines_start:.4f}s"
            )

        add_polylines_start = time.time()
        self.add_polylines(polylines)
        logger.info(f"Added polylines in {time.time() - add_polylines_start:.4f}s")
        logger.info(f"Total CustomSkeleton init time: {time.time() - start_time:.4f}s")

    def _get_vertex_index(self, vertex):
        if type(vertex) is not tuple:
            vertex = tuple(vertex)
        return self.vertices.index(tuple(vertex))

    def add_vertex(self, vertex, radius=None):
        if type(vertex) is not tuple:
            vertex = tuple(vertex)

        # if vertex not in self.vertices:
        self.vertices.append(vertex)
        if radius:
            self.radii.append(radius)
        # else:
        #    print(f"Vertex {vertex} already in skeleton, not adding")

    def add_vertices(self, vertices, radii):
        if radii:
            for vertex, radius in zip(vertices, radii):
                self.add_vertex(vertex, radius)
        else:
            for vertex in vertices:
                self.add_vertex(vertex)
        self.vertices = self.vertices

    def add_edge(self, edge):
        if not isinstance(edge[0], (int, np.integer)):
            # then edges are coordinates, so need to get corresponding radii
            edge_start_id = self._get_vertex_index(edge[0])
            edge_end_id = self._get_vertex_index(edge[1])
            edge = (edge_start_id, edge_end_id)
        self.edges.append(edge)

    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge)

    def add_polylines(self, polylines):
        for polyline in polylines:
            self.add_polyline(polyline)

    def add_polyline(self, polyline):
        self.polylines.append(polyline)

    def simplify(self, tolerance_nm=200):
        """
        Simplify skeleton by removing vertices close to straight lines (RDP, per polyline
        between branch points), without losing any of its loops.
        """
        start_time = time.time()
        logger.info(f"Starting simplification with tolerance {tolerance_nm}nm")
    
        if not self.polylines:
            g = self.skeleton_to_graph()
            polylines = self.get_polylines_positions_from_graph(g)
        else:
            polylines = self.polylines

        simplified_polylines = []
        all_vertices_set = set()

        # Use consistent rounding precision (e.g., 6 decimal places)
        PRECISION = 6

        polylines = [np.asarray(polyline, dtype=float) for polyline in polylines]
        keeps = []
        for points in polylines:
            keep = np.ones(len(points), dtype=bool)
            if len(points) > 2:
                keep = rdp(points, epsilon=tolerance_nm, return_mask=True).astype(bool)
                keep[[0, -1]] = True
            keeps.append(keep)
        CustomSkeleton._keep_cycles(polylines, keeps)

        for points, keep in zip(polylines, keeps):
            simplified_rounded = [
                tuple(round(float(x), PRECISION) for x in vertex) for vertex in points[keep]
            ]
            all_vertices_set.update(simplified_rounded)
            simplified_polylines.append(simplified_rounded)
    
        logger.info(f"RDP produced {len(all_vertices_set)} unique vertices")
    
        # Create vertex mapping
        vertices_list = list(all_vertices_set)
        vertex_to_idx = {v: i for i, v in enumerate(vertices_list)}
    
        # Build edges - should never fail now
        edges = []
        for polyline in simplified_polylines:
            indices = [vertex_to_idx[v] for v in polyline]
            
            if len(indices) > 1:
                edges.extend(list(zip(indices[:-1], indices[1:])))
    
        logger.info(f"Built {len(edges)} edges from {len(simplified_polylines)} polylines")
    
        # Handle radii if present
        radii = None
        if self.radii:
            orig_vertex_to_idx = {}
            for i, v in enumerate(self.vertices):
                v_rounded = tuple(round(float(x), PRECISION) for x in v)
                orig_vertex_to_idx[v_rounded] = i
            
            radii = []
            for v in vertices_list:
                if v in orig_vertex_to_idx:
                    radii.append(self.radii[orig_vertex_to_idx[v]])
                else:
                    radii.append(0.0)
    
        result = CustomSkeleton(vertices_list, edges, radii, simplified_polylines)
        logger.info(f"Total simplification time: {time.time() - start_time:.4f}s")

        return result

    @staticmethod
    def _keep_cycles(polylines, keeps):
        """Restore just enough of the vertices RDP dropped that no cycle is lost.

        RDP keeps each polyline within the tolerance of the original, but alone it can
        change the skeleton's topology. A closed polyline narrower than about twice the
        tolerance shrinks to the line to its farthest vertex and back; parallel polylines
        between the same two branch points, each within the tolerance of the straight
        line between those points, all shrink to that one line. Either way a cycle of the
        skeleton disappears. Restored vertices are original ones, farthest first, and
        polylines that keep their cycles are left exactly as RDP made them.
        """
        bare = {}
        for i, (points, keep) in enumerate(zip(polylines, keeps)):
            if len(points) > 3 and np.allclose(points[0], points[-1]):
                # a closed polyline stays a cycle with three distinct vertices
                while len(np.unique(points[keep], axis=0)) < 3:
                    if not CustomSkeleton._restore_farthest_vertex(points, keep):
                        break
            elif keep.sum() == 2 and len(points) >= 2:
                ends = tuple(sorted(tuple(np.round(p, 6)) for p in (points[0], points[-1])))
                bare.setdefault(ends, []).append(i)

        for group in bare.values():
            # Only one of these may stay the bare line between the two branch points: a
            # single original edge if there is one (it has no vertex to restore), else
            # the straightest.
            group.sort(
                key=lambda i: (
                    len(polylines[i]) > 2,
                    CustomSkeleton._max_chord_distance(polylines[i]),
                )
            )
            for i in group[1:]:
                CustomSkeleton._restore_farthest_vertex(polylines[i], keeps[i])

    @staticmethod
    def _restore_farthest_vertex(points, keep):
        """Keep the dropped vertex farthest from the simplified polyline, measured from the
        kept segment it was dropped from; False when none is dropped."""
        kept = np.flatnonzero(keep)
        best, best_distance = None, -1.0
        for a, b in zip(kept[:-1], kept[1:]):
            if b - a > 1:
                inner = np.arange(a + 1, b)
                distances = CustomSkeleton.lineseg_dists(points[inner], points[a], points[b])
                j = int(np.argmax(distances))
                if distances[j] > best_distance:
                    best, best_distance = inner[j], distances[j]
        if best is None:
            return False
        keep[best] = True
        return True

    @staticmethod
    def _max_chord_distance(points):
        if len(points) < 3:
            return 0.0
        return float(np.max(CustomSkeleton.lineseg_dists(points[1:-1], points[0], points[-1])))

    @staticmethod
    def find_branchpoints_and_endpoints(graph):
        start_time = time.time()
        branchpoints = []
        endpoints = []
        # print("graph.nodes", len(graph.nodes))
        for node in graph.nodes:
            degree = graph.degree[node]
            if degree <= 1:
                endpoints.append(node)
            elif degree > 2:
                branchpoints.append(node)

        logger.info(
            f"Found {len(branchpoints)} branchpoints and {len(endpoints)} endpoints in {time.time() - start_time:.4f}s"
        )
        return branchpoints, endpoints

    @staticmethod
    def get_polyline_from_subgraph(subgraph, all_graph_edges):
        if len(subgraph.nodes) == 1:
            # then contains a single node
            node = list(subgraph.nodes)[0]
            path = [(node, node)]
        else:
            path = list(nx.eulerian_path(subgraph))
        start_node = path[0][0]
        end_node = path[-1][-1]
        prepended = False
        appended = False
        output_path = path.copy()
        for edge in all_graph_edges:
            if edge not in path and edge[::-1] not in path:
                if start_node in edge and not prepended:
                    output_path.insert(0, edge if edge[1] == start_node else edge[::-1])
                    prepended = True
                elif end_node in edge and not appended:
                    output_path.append(edge if edge[0] == end_node else edge[::-1])
                    appended = True

        if len(output_path) == 1 and output_path[0][0] == output_path[0][1]:
            # then single node
            polyline = [output_path[0][0]]
        else:
            # remove edge in output_path if the start and endpoints are the same, which we added above if there is only one endpoint
            for edge in output_path:
                if edge[0] == edge[1]:
                    output_path.remove(edge)

            polyline = [node for node, _ in output_path]
            polyline.append(output_path[-1][-1])

        return polyline

    @staticmethod
    def get_polylines_from_graph(g):
        """Extract polylines as paths between branch/end points"""
        import time
        t0 = time.time()
        
        # Find special nodes (degree != 2)
        branchpoints, endpoints = CustomSkeleton.find_branchpoints_and_endpoints(g)
        special_nodes = set(branchpoints) | set(endpoints)
        
        t1 = time.time()
        logger.info(f"Found {len(branchpoints)} branchpoints and {len(endpoints)} endpoints in {t1-t0:.4f}s")
        
        polylines = []
        visited_edges = set()
        
        # Start from each special node and traverse until hitting another special node
        for start_node in special_nodes:
            for neighbor in g.neighbors(start_node):
                edge = tuple(sorted([start_node, neighbor]))
                
                if edge in visited_edges:
                    continue
                    
                # Trace polyline from start_node through neighbor
                polyline = [start_node]
                current = neighbor
                prev = start_node
                
                visited_edges.add(edge)
                
                # Follow chain of degree-2 nodes
                while current not in special_nodes:
                    polyline.append(current)
                    
                    # Find next node (the neighbor that isn't prev)
                    neighbors = list(g.neighbors(current))
                    next_nodes = [n for n in neighbors if n != prev]
                    
                    if len(next_nodes) != 1:
                        # Shouldn't happen if special_nodes is correct
                        logger.warning(f"Node {current} has {len(next_nodes)} next nodes (degree={g.degree(current)})")
                        break
                        
                    next_node = next_nodes[0]
                    edge = tuple(sorted([current, next_node]))

                    if edge in visited_edges:
                        # We've hit a cycle or already processed edge
                        # Check if this creates a loop back to start
                        if next_node == start_node:
                            # This is a loop - mark it by appending start again
                            polyline.append(current)
                            polyline.append(start_node)
                            polylines.append(polyline)
                            visited_edges.add(edge)
                        break

                    visited_edges.add(edge)
                    prev = current
                    current = next_node

                # Add final special node (if we didn't break due to loop)
                if len(polyline) == 1 or polyline[-1] != start_node:
                    polyline.append(current)
                    polylines.append(polyline)
        
        t2 = time.time()
        logger.info(f"Extracted {len(polylines)} polylines in {t2-t1:.4f}s")
        
        # Check for missing edges (isolated cycles with no special nodes)
        all_edges = set(tuple(sorted(edge)) for edge in g.edges)
        if len(visited_edges) < len(all_edges):
            missing_edges = all_edges - visited_edges
            logger.info(f"Found {len(missing_edges)} edges in isolated cycles")

            # Handle cycles separately
            g_remaining = g.edge_subgraph(missing_edges).copy()
            for component in nx.connected_components(g_remaining):
                if len(component) > 1:
                    # It's a cycle - create polyline from arbitrary start
                    g_sub = g_remaining.subgraph(component)
                    start = list(component)[0]
                    cycle = [start]
                    current = start
                    prev = None

                    while True:
                        neighbors = [n for n in g_sub.neighbors(current) if n != prev]
                        if not neighbors:
                            break
                        next_node = neighbors[0]
                        if next_node == start and len(cycle) > 2:
                            break
                        cycle.append(next_node)
                        prev = current
                        current = next_node

                    # Mark as loop by appending first vertex to close the cycle
                    cycle.append(start)
                    polylines.append(cycle)
        
        t3 = time.time()
        logger.info(f"Total time: {t3-t0:.4f}s, extracted {len(polylines)} polylines")
        return polylines  # <-- This was missing!


    @staticmethod
    def get_polyline_from_subgraph_fast(subgraph, all_graph_edges_set):
        """Optimized version that returns edges too"""
        if len(subgraph.nodes) == 1:
            node = list(subgraph.nodes)[0]
            return [node], set()

        if not nx.is_eulerian(subgraph):
            # Handle non-eulerian case
            # For non-Eulerian graphs, we can't traverse all edges exactly once
            # But we should still return the edges properly
            # The polyline will just be a minimal representation
            nodes = list(subgraph.nodes)
            if len(nodes) == 0:
                return [], set()

            # Find a path through the graph if possible (DFS or BFS)
            # Start from any node
            start_node = nodes[0]

            # Try to find the longest simple path using DFS
            # For efficiency, just do a DFS traversal
            visited = set()
            polyline = []

            def dfs(node):
                visited.add(node)
                polyline.append(node)
                for neighbor in subgraph.neighbors(node):
                    if neighbor not in visited:
                        dfs(neighbor)

            dfs(start_node)

            # Collect all edges from the subgraph
            polyline_edges = {
                tuple(sorted([u, v])) for u, v in subgraph.edges
            }
            return polyline, polyline_edges

        path = list(nx.eulerian_path(subgraph))

        if not path:
            nodes = list(subgraph.nodes)
            return nodes[:1], set()

        start_node = path[0][0]
        end_node = path[-1][-1]
        output_path = path.copy()

        # Optimized edge checking
        for edge in all_graph_edges_set:
            if edge in path or edge[::-1] in path:
                continue
            if start_node in edge:
                output_path.insert(0, edge if edge[1] == start_node else edge[::-1])
                break

        for edge in all_graph_edges_set:
            if edge in path or edge[::-1] in path:
                continue
            if end_node in edge:
                output_path.append(edge if edge[0] == end_node else edge[::-1])
                break

        # Build polyline
        if len(output_path) == 1 and output_path[0][0] == output_path[0][1]:
            return [output_path[0][0]], set()

        # Remove self-loops
        output_path = [edge for edge in output_path if edge[0] != edge[1]]

        if not output_path:
            return [start_node], set()

        polyline = [node for node, _ in output_path]
        polyline.append(output_path[-1][-1])

        # Create edge set efficiently
        polyline_edges = {
            tuple(sorted([polyline[i], polyline[i + 1]]))
            for i in range(len(polyline) - 1)
        }

        return polyline, polyline_edges

    @staticmethod
    def remove_smallest_qualifying_branch(g, min_branch_length_nm=200):
        start_time = time.time()
        logger.info(f"Searching for branches shorter than {min_branch_length_nm}nm")

        # get endpoints and branchpoints from g
        branchpoints_start = time.time()
        branchpoints, _ = CustomSkeleton.find_branchpoints_and_endpoints(g)
        logger.info(f"Found branchpoints in {time.time() - branchpoints_start:.4f}s")

        current_min_branch_length_nm = np.inf
        current_min_branch_path = None
        # for endpoint in endpoints:
        #     for branchpoint in branchpoints:
        #         path_length_nm = nx.shortest_path_length(
        #             g, endpoint, branchpoint, weight="weight"
        #         )
        #         if (
        #             path_length_nm < min_tick_length_nm
        #             and path_length_nm < current_min_tick_length_nm
        #         ):
        #             path = nx.dijkstra_path(g, endpoint, branchpoint, weight="weight")
        #             if len(path) < g.number_of_nodes():
        #                 current_min_tick_length_nm = path_length_nm
        #                 current_min_tick_path = path

        polylines_start = time.time()
        polylines_by_vertex_id = CustomSkeleton.get_polylines_from_graph(g)
        logger.info(
            f"Got {len(polylines_by_vertex_id)} polylines in {time.time() - polylines_start:.4f}s"
        )

        search_start = time.time()
        branches_checked = 0
        for polyline_by_vertex_id in polylines_by_vertex_id:
            if (polyline_by_vertex_id[0] in branchpoints) ^ (
                polyline_by_vertex_id[-1] in branchpoints
            ):
                branches_checked += 1
                polyline_length_nm = 0
                for v1, v2 in zip(
                    polyline_by_vertex_id[:-1], polyline_by_vertex_id[1:]
                ):
                    polyline_length_nm += np.linalg.norm(
                        np.array(g.nodes[v1]["position_nm"])
                        - np.array(g.nodes[v2]["position_nm"])
                    )
                if (
                    polyline_length_nm < min_branch_length_nm
                    and polyline_length_nm < current_min_branch_length_nm
                ):
                    if len(set(polyline_by_vertex_id)) < g.number_of_nodes():
                        current_min_branch_length_nm = polyline_length_nm
                        current_min_branch_path = polyline_by_vertex_id
        logger.info(
            f"Checked {branches_checked} branches in {time.time() - search_start:.4f}s"
        )

        if current_min_branch_path:
            remove_start = time.time()
            g.remove_edges_from(
                list(zip(current_min_branch_path[:-1], current_min_branch_path[1:]))
            )
            g.remove_nodes_from(list(nx.isolates(g)))
            logger.info(
                f"Removed branch of length {current_min_branch_length_nm:.2f}nm in {time.time() - remove_start:.4f}s"
            )
        else:
            logger.info("No qualifying branch found to remove")

        logger.info(
            f"Total remove_smallest_qualifying_branch time: {time.time() - start_time:.4f}s"
        )
        return current_min_branch_path, g

    def skeleton_to_graph(self):
        start_time = time.time()
        logger.info(
            f"Converting skeleton to graph: {len(self.vertices)} vertices, {len(self.edges)} edges"
        )

        graph_init_start = time.time()
        g = nx.Graph()
        g.add_nodes_from(range(len(self.vertices)))
        logger.info(
            f"Initialized graph with nodes in {time.time() - graph_init_start:.4f}s"
        )

        node_attrs_start = time.time()
        for idx in range(len(self.vertices)):
            g.nodes[idx]["position_nm"] = self.vertices[idx]
            # add radii as properties to the nodes
            if self.radii:
                g.nodes[idx]["radius"] = self.radii[idx]
        logger.info(f"Added node attributes in {time.time() - node_attrs_start:.4f}s")

        edges_start = time.time()
        g.add_edges_from(self.edges)
        logger.info(f"Added edges in {time.time() - edges_start:.4f}s")

        # add edge weights to the graph where weights are the distances between vertices
        weights_start = time.time()
        for edge in self.edges:
            try:
                g[edge[0]][edge[1]]["weight"] = np.linalg.norm(
                    np.array(self.vertices[edge[0]]) - np.array(self.vertices[edge[1]])
                )
            except IndexError as e:
                logger.error(
                    f"IndexError when trying to access vertices for edge {edge} with vertices length {len(self.vertices)}"
                )
                raise e
        logger.info(f"Calculated edge weights in {time.time() - weights_start:.4f}s")
        logger.info(f"Total skeleton_to_graph time: {time.time() - start_time:.4f}s")

        return g

    @staticmethod
    def get_polylines_positions_from_graph(g):
        start_time = time.time()
        logger.info("Extracting polyline positions from graph")

        polylines_start = time.time()
        polylines_by_vertex_id = CustomSkeleton.get_polylines_from_graph(g)
        logger.info(
            f"Got polylines by vertex ID in {time.time() - polylines_start:.4f}s"
        )

        positions_start = time.time()
        polylines = []
        for polyline_by_vertex_id in polylines_by_vertex_id:
            polylines.append(
                np.array(
                    [
                        np.array(g.nodes[vertex_id]["position_nm"])
                        for vertex_id in polyline_by_vertex_id
                    ]
                )
            )
        logger.info(f"Converted to positions in {time.time() - positions_start:.4f}s")
        logger.info(
            f"Total get_polylines_positions_from_graph time: {time.time() - start_time:.4f}s"
        )

        return polylines

    def graph_to_skeleton(self, g, preserve_polylines=False):
        start_time = time.time()
        logger.info(f"Converting graph to skeleton with {g.number_of_nodes()} nodes")

        vertices_start = time.time()
        vertices = [self.vertices[idx] for idx in g.nodes]
        radii = [self.radii[idx] for idx in g.nodes] if self.radii else None
        logger.info(
            f"Extracted {len(vertices)} vertices and radii in {time.time() - vertices_start:.4f}s"
        )

        edges_start = time.time()
        edges = fastremap.remap(
            np.array(g.edges), dict(zip(list(g.nodes), list(range(len(g.nodes)))))
        )
        edges = edges.tolist()
        logger.info(f"Remapped {len(edges)} edges in {time.time() - edges_start:.4f}s")

        polylines_start = time.time()
        if preserve_polylines and self.polylines:
            # Filter existing polylines to only include vertices that remain in the graph
            logger.info("Preserving original polylines, filtering removed vertices")
            remaining_vertices = set(self.vertices[idx] for idx in g.nodes)
            polylines = []
            for polyline in self.polylines:
                # Filter out vertices that were removed
                filtered_polyline = [v for v in polyline if tuple(v) in remaining_vertices]
                if len(filtered_polyline) > 1:
                    polylines.append(np.array(filtered_polyline))
            logger.info(f"Filtered polylines: {len(self.polylines)} -> {len(polylines)}")
        else:
            polylines = CustomSkeleton.get_polylines_positions_from_graph(g)
            logger.info(f"Extracted new polylines from graph")
        logger.info(f"Polyline processing took {time.time() - polylines_start:.4f}s")

        skeleton_start = time.time()
        skeleton = CustomSkeleton(vertices, edges, radii, polylines)
        logger.info(f"Created skeleton in {time.time() - skeleton_start:.4f}s")
        logger.info(f"Total graph_to_skeleton time: {time.time() - start_time:.4f}s")

        return skeleton

    def prune(self, min_branch_length_nm=200, max_iterations=100):
        """
        Smart iterative pruning - only traces terminal branches.
        Avoids expensive full polyline extraction every iteration.
        """
        if len(self.vertices) == 1:
            return self

        g = self.skeleton_to_graph()
        total_removed = 0

        for iteration in range(max_iterations):
            # Find all terminal branches by tracing from endpoints
            terminal_branches = self._find_all_terminal_branches(
                g, min_branch_length_nm
            )

            if not terminal_branches:
                logger.info(
                    f"Pruning converged after {iteration} iterations, removed {total_removed} branches total"
                )
                break

            # Remove all short terminal branches
            for branch_vertices in terminal_branches:
                # branch_vertices already excludes the branchpoint
                # Just remove these nodes - NetworkX will handle the edges automatically
                g.remove_nodes_from(branch_vertices)
            
            total_removed += len(terminal_branches)
            logger.info(
                f"Iteration {iteration}: removed {len(terminal_branches)} branches ({total_removed} total)"
            )

        else:
            logger.warning(
                f"Pruning did not converge after {max_iterations} iterations"
            )

        return self.graph_to_skeleton(g, preserve_polylines=False)

    def prune_terminal_branches(
        self,
        min_branch_length_nm=100.0,
        radius_scale=1.5,
        width_scale=1.5,
        width_anisotropy=2.0,
        mask=None,
        mask_origin_nm=None,
        voxel_size_nm=None,
    ):
        """Remove terminal branches that do not stick out of the body they
        grow from.

        A terminal branch runs from a tip (degree 1) to the junction
        (degree >= 3) it meets. With ``L`` its length and ``r`` the junction's
        radius (the EDT value stored on the vertex), it is removed when

            L <= radius_scale * r + min_branch_length_nm       (radius rule)

        That is the scale axis criterion on the skeleton's inscribed balls:
        the tip's ball is swallowed by the junction's ball inflated by
        ``radius_scale``, so the branch is a bump of the body, not an arm.
        ``min_branch_length_nm`` is the absolute floor (mesh-n-bone's prune).

        The radius rule cannot see the teeth that thinning leaves on a sheet
        (there ``r`` is half the sheet thickness). So where the cross-section
        around a branch is sheet-like, ``W_max / W_min >= width_anisotropy``
        with ``W_max``/``W_min`` the longest/shortest distance from the branch
        to the surface along rays perpendicular to it, the branch is also
        removed when

            L <= width_scale * W_max                            (width rule)

        A round lobe (``W_max ~ W_min``) is judged by the radius rule alone.
        The two ends of each component's longest shortest path are exempt from
        the width rule, so a sheet keeps its axis.

        Branches are removed one at a time, smallest ratio first, and all
        terminal branches are re-measured after each removal (removing an arm
        can turn a junction into a pass-through point). A bare path or a ring
        has no junction and is returned unchanged.

        ``radius_scale=0`` disables the radius rule (leaving the absolute
        floor), ``width_scale=0`` disables the width rule, and
        ``width_anisotropy=None`` applies the width rule to every branch. The
        width rule needs ``mask`` (bool, indexed z, y, x), ``mask_origin_nm``
        (x, y, z centre of voxel (0, 0, 0)) and ``voxel_size_nm`` (scalar or
        x, y, z); the radius rule needs ``self.radii``.
        """
        if len(self.vertices) <= 1:
            return self
        if radius_scale > 0 and not self.radii:
            raise ValueError("radius_scale > 0 needs per-vertex radii on the skeleton")
        use_width = width_scale > 0
        if use_width and (mask is None or mask_origin_nm is None or voxel_size_nm is None):
            raise ValueError(
                "width_scale > 0 needs mask, mask_origin_nm and voxel_size_nm"
            )

        g = self.skeleton_to_graph()
        if not any(d >= 3 for _, d in g.degree()):
            return self

        half_voxel = float(np.min(voxel_size_nm)) / 2 if use_width else 0.0
        widths = {}
        ends = axis_ends(g) if use_width else set()
        removed = 0
        while True:
            best = None
            for branch, junction in terminal_branches(g):
                length = branch_length(g, branch, junction)
                r = float(g.nodes[junction].get("radius", 0.0))
                q = length / (radius_scale * r + min_branch_length_nm + 1e-9)
                if q > 1 and use_width and branch[0] not in ends:
                    key = (branch[0], junction, len(branch))
                    if key not in widths:
                        widths[key] = branch_width(
                            g, branch, junction, mask, mask_origin_nm, voxel_size_nm
                        )
                    w_max, w_min = widths[key]
                    sheet_like = (
                        width_anisotropy is None
                        or width_anisotropy <= 0
                        or w_max / max(w_min, half_voxel) >= width_anisotropy
                    )
                    if sheet_like:
                        q = min(q, length / (width_scale * w_max + 1e-9))
                if q <= 1 and (best is None or q < best[0]):
                    best = (q, branch)
            if best is None:
                break
            g.remove_nodes_from(best[1])
            removed += 1
            if use_width and ends.intersection(best[1]):
                ends = axis_ends(g)
        logger.info(f"prune_terminal_branches removed {removed} branches")
        return self.graph_to_skeleton(g, preserve_polylines=False)

    def _find_all_terminal_branches(self, g, min_branch_length_nm):
        """
        Find all terminal branches shorter than threshold.
        Only traces from endpoints to first branchpoint - much faster!
        """
        branchpoints, endpoints = CustomSkeleton.find_branchpoints_and_endpoints(g)

        if not branchpoints or not endpoints:
            return []

        terminal_branches = []

        # For each endpoint, trace path to nearest branchpoint
        for endpoint in endpoints:
            visited = {endpoint}
            current = endpoint
            path = [endpoint]

            while True:
                neighbors = [n for n in g.neighbors(current) if n not in visited]

                if not neighbors:
                    break  # Isolated endpoint

                if len(neighbors) > 1:
                    break  # Shouldn't happen in proper tree traversal

                next_node = neighbors[0]
                visited.add(next_node)
                path.append(next_node)

                if next_node in branchpoints:
                    # Reached branchpoint - calculate total path length
                    # DON'T include the branchpoint in the path to remove
                    path_length = self.calculate_polyline_length_vectorized(path, g)

                    if path_length < min_branch_length_nm:
                        terminal_branches.append(path[:-1])  # ← Exclude the branchpoint!
                    break

                current = next_node

        return terminal_branches

    @staticmethod
    def calculate_polyline_length_vectorized(polyline_vertices, g):
        """Vectorized length calculation"""
        if len(polyline_vertices) < 2:
            return 0.0

        positions = np.array([g.nodes[v]["position_nm"] for v in polyline_vertices])
        diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)

    @staticmethod
    def lineseg_dists(p, a, b):
        # https://stackoverflow.com/questions/54442057/calculate-the-euclidian-distance-between-an-array-of-points-to-a-line-segment-in/54442561#54442561
        # Handle case where p is a single point, i.e. 1d array.
        p = np.atleast_2d(p)

        # TODO for you: consider implementing @Eskapp's suggestions
        if np.all(a == b):
            return np.linalg.norm(p - a, axis=1)

        # normalized tangent vector
        d = np.divide(b - a, np.linalg.norm(b - a))

        # signed parallel distance components
        s = np.dot(a - p, d)
        t = np.dot(p - b, d)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, np.zeros_like(s)])

        # perpendicular distance component, as before
        # note that for the 3D case these will be vectors
        c = np.linalg.norm(np.cross(p - a, d), axis=1)

        # use hypot for Pythagoras to improve accuracy
        return np.hypot(h, c)

    def encode_neuroglancer_bytes(self, include_radii=True) -> bytes:
        """Return the bytes that would be written to disk by
        write_neuroglancer_skeleton, without touching the filesystem.

        Empty skeletons are encoded as 8 bytes (two zero uint32s), matching
        the on-disk format used by neuroglancer for an empty chunk.

        When ``include_radii`` is set and this skeleton carries a per-vertex
        ``radii`` array aligned with its vertices, the radii are written as a
        single-component ``radius`` vertex attribute (float32). The skeleton
        source's ``info`` must declare a matching ``vertex_attributes`` entry,
        and every skeleton in that source must carry it -- so callers writing
        to a source that declares the attribute must pass radii (or an empty
        skeleton, which has no vertices and therefore no attribute values).
        """
        import struct

        if len(self.vertices) == 0:
            return struct.pack("<II", 0, 0)

        # neuroglancer requires edges as an ndarray of shape (N, 2).
        # CustomSkeleton.add_edges() turns the input into a Python list, so
        # coerce here to protect single-vertex / zero-edge cases (seed-voxel
        # fallback, etc.).
        if len(self.edges) == 0:
            edges = np.zeros((0, 2), dtype=np.uint32)
        else:
            edges = np.asarray(self.edges, dtype=np.uint32)

        vertex_attributes = None
        source = Source()
        if include_radii and self.radii is not None and len(self.radii) == len(
            self.vertices
        ):
            radii = np.asarray(self.radii, dtype="<f4")
            vertex_attributes = {"radius": radii}
            source = Source()
            source.vertex_attributes = {
                "radius": VertexAttributeInfo(
                    data_type=np.dtype("float32"), num_components=1
                )
            }

        skel = NeuroglancerSkeleton(
            self.vertices, edges, vertex_attributes=vertex_attributes
        )
        return skel.encode(source)

    def write_neuroglancer_skeleton(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        encoded = self.encode_neuroglancer_bytes()
        with open(path, "wb") as f:
            f.write(encoded)

    @staticmethod
    def read_neuroglancer_skeleton(path):
        start_time = time.time()
        logger.info(f"Reading Neuroglancer skeleton from {path}")

        read_start = time.time()
        source_info = Source()
        with open(path, "rb") as f:
            data = f.read()
        logger.info(f"Read {len(data)} bytes in {time.time() - read_start:.4f}s")

        offset = 0

        # 1) Read number of vertices (n_vertices) and number of edges (n_edges).
        header_start = time.time()
        n_vertices, n_edges = struct.unpack_from("<II", data, offset)
        offset += 8
        logger.info(
            f"Read header: {n_vertices} vertices, {n_edges} edges in {time.time() - header_start:.4f}s"
        )

        # 2) Decode vertex_positions.
        vertices_start = time.time()
        num_vp_values = n_vertices * 3
        vertex_positions = np.frombuffer(
            data, dtype="<f4", count=num_vp_values, offset=offset
        )
        offset += vertex_positions.nbytes
        # Reshape to (n_vertices, vp_dim).
        vertex_positions = vertex_positions.reshape((n_vertices, 3))
        logger.info(f"Decoded vertex positions in {time.time() - vertices_start:.4f}s")

        # 3) Decode edges.
        edges_start = time.time()
        num_edge_values = n_edges * 2
        edges = np.frombuffer(data, dtype="<u4", count=num_edge_values, offset=offset)
        offset += edges.nbytes
        # Reshape to (n_edges, edges_dim).
        edges = edges.reshape((n_edges, 2))
        logger.info(f"Decoded edges in {time.time() - edges_start:.4f}s")

        # 4) Decode vertex_attributes (if any).
        attrs_start = time.time()
        decoded_attributes = {}
        if source_info.vertex_attributes:
            for attr_name, (
                attr_dtype,
                num_components,
            ) in source_info.vertex_attributes.items():
                # We expect shape = (n_vertices, num_components).
                expected_size = n_vertices * num_components
                attribute = np.frombuffer(
                    data,
                    dtype=attr_dtype.newbyteorder("<"),
                    count=expected_size,
                    offset=offset,
                )
                offset += attribute.nbytes
                attribute = attribute.reshape((n_vertices, num_components))
                decoded_attributes[attr_name] = attribute
            logger.info(
                f"Decoded {len(decoded_attributes)} vertex attributes in {time.time() - attrs_start:.4f}s"
            )

        logger.info(
            f"Total read_neuroglancer_skeleton time: {time.time() - start_time:.4f}s"
        )

        # 5) Package results.
        return vertex_positions, edges


# ---------------------------------------------------------------------------
# Terminal-branch pruning by the scale of the body a branch grows from.
# Used by CustomSkeleton.prune_terminal_branches; see its docstring.
# ---------------------------------------------------------------------------

_RAY_AZIMUTHS = np.radians(np.arange(0, 360, 15))  # 24 rays around a point
_RAY_STEP_VOXELS = 0.5
_RAY_MAX_STEPS = 600  # 300 voxels


def terminal_branches(g):
    """Yield ``(branch, junction)`` for every terminal branch of ``g``: the
    nodes from a tip (degree 1) up to, but excluding, the first junction
    (degree >= 3). A tip whose walk ends at another tip (a bare path) yields
    nothing: a path has no junction to judge it against."""
    for tip in [n for n, d in g.degree() if d == 1]:
        path, prev, cur = [tip], None, tip
        while True:
            nxt = [m for m in g.neighbors(cur) if m != prev]
            if len(nxt) != 1:
                break
            prev, cur = cur, nxt[0]
            if g.degree[cur] >= 3:
                yield path, cur
                break
            path.append(cur)


def branch_length(g, branch, junction):
    """Length (nm) of a terminal branch from its tip to the junction."""
    path = branch + [junction]
    return float(sum(g.edges[a, b]["weight"] for a, b in zip(path, path[1:])))


def axis_ends(g):
    """Tips of the longest shortest path of every component with >= 2 nodes."""
    ends = set()
    for comp in nx.connected_components(g):
        if len(comp) < 2:
            continue
        sub = g.subgraph(comp)
        d = nx.single_source_dijkstra_path_length(sub, next(iter(comp)), weight="weight")
        a = max(d, key=d.get)
        d = nx.single_source_dijkstra_path_length(sub, a, weight="weight")
        ends.update((a, max(d, key=d.get)))
    return ends


def _perpendicular_basis(u):
    a = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def ray_extent(mask, points, direction):
    """Distance (voxels) from each point to the boundary of ``mask`` along rays
    perpendicular to ``direction``: ``(longest, shortest)`` over 24 azimuths.
    ``points`` are float voxel coordinates in the axis order of ``mask``."""
    e1, e2 = _perpendicular_basis(np.asarray(direction, dtype=float))
    dirs = np.outer(np.cos(_RAY_AZIMUTHS), e1) + np.outer(np.sin(_RAY_AZIMUTHS), e2)
    steps = np.arange(1, _RAY_MAX_STEPS + 1) * _RAY_STEP_VOXELS
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    longest, shortest = [], []
    for chunk in np.array_split(points, max(1, int(np.ceil(len(points) / 32)))):
        pos = chunk[:, None, None, :] + steps[None, None, :, None] * dirs[None, :, None, :]
        idx = np.rint(pos).astype(np.int64)
        inside = np.ones(idx.shape[:3], dtype=bool)
        for ax in range(3):
            inside &= (idx[..., ax] >= 0) & (idx[..., ax] < mask.shape[ax])
            idx[..., ax] = np.clip(idx[..., ax], 0, mask.shape[ax] - 1)
        out = ~(inside & mask[idx[..., 0], idx[..., 1], idx[..., 2]])
        first = np.where(out.any(axis=2), out.argmax(axis=2), _RAY_MAX_STEPS) * _RAY_STEP_VOXELS
        longest.append(first.max(axis=1))
        shortest.append(first.min(axis=1))
    return np.concatenate(longest), np.concatenate(shortest)


def branch_width(g, branch, junction, mask, mask_origin_nm, voxel_size_nm):
    """``(W_max, W_min)`` in nm for a terminal branch: the median, over the tip
    half of the branch's nodes, of the longest and the shortest distance from
    the node to the object's surface along rays perpendicular to the branch.
    ``mask`` is indexed (z, y, x); node positions, ``mask_origin_nm`` (the
    centre of voxel (0, 0, 0)) and ``voxel_size_nm`` are (x, y, z)."""
    tip_half = branch[: max(1, len(branch) // 2)]
    origin = np.asarray(mask_origin_nm, dtype=float)
    vs = np.broadcast_to(np.asarray(voxel_size_nm, dtype=float), (3,))
    pts = (np.array([g.nodes[n]["position_nm"] for n in tip_half], dtype=float) - origin) / vs
    u = np.asarray(g.nodes[branch[0]]["position_nm"], dtype=float) - np.asarray(
        g.nodes[junction]["position_nm"], dtype=float
    )
    u = u / (np.linalg.norm(u) + 1e-9)
    longest, shortest = ray_extent(mask, pts[:, ::-1], u[::-1])
    scale = float(np.min(vs))
    return float(np.median(longest)) * scale, float(np.median(shortest)) * scale

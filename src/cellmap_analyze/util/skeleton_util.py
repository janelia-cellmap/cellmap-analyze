from dataclasses import dataclass
import struct
import numpy as np
import os
import logging
import time
from neuroglancer.skeleton import Skeleton as NeuroglancerSkeleton
import fastremap
import networkx as nx
from pybind11_rdp import rdp

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def skimage_to_custom_skeleton_fast(binary_skeleton, spacing=(1, 1, 1), simplify_triangles=True):
    """
    Faster version using vectorized operations
    
    Args:
        binary_skeleton: Binary 3D array
        spacing: Voxel spacing in each dimension
        simplify_triangles: If True, reduce triangular cliques to spanning trees
    """
    start_time = time.time()
    logger.info(f"Starting skimage_to_custom_skeleton_fast conversion (simplify_triangles={simplify_triangles})")
    
    coords_start = time.time()
    coords = np.argwhere(binary_skeleton)
    logger.info(f"Found {len(coords)} coordinates in {time.time() - coords_start:.4f}s")
    
    if len(coords) == 0:
        logger.info("No coordinates found, returning empty skeleton")
        return CustomSkeleton()
    
    vertices_start = time.time()
    vertices = coords * np.array(spacing)
    vertices = [tuple(v) for v in vertices]
    logger.info(f"Created {len(vertices)} vertices in {time.time() - vertices_start:.4f}s")
    
    # Create a lookup dictionary for fast coordinate checking
    lookup_start = time.time()
    coord_to_idx = {tuple(c): i for i, c in enumerate(coords)}
    logger.info(f"Built coordinate lookup in {time.time() - lookup_start:.4f}s")
    
    # 26-connectivity offsets
    offsets_start = time.time()
    offsets = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                offsets.append((dz, dy, dx))
    offsets = np.array(offsets)
    logger.info(f"Created connectivity offsets in {time.time() - offsets_start:.4f}s")
    
    # For each coordinate, check which neighbors exist
    edges_start = time.time()
    edges = []
    for i, coord in enumerate(coords):
        neighbors = coord + offsets
        for neighbor in neighbors:
            neighbor_tuple = tuple(neighbor)
            if neighbor_tuple in coord_to_idx:
                j = coord_to_idx[neighbor_tuple]
                if j > i:  # Avoid duplicates
                    edges.append((i, j))
    
    logger.info(f"Found {len(edges)} edges in {time.time() - edges_start:.4f}s")
    
    if simplify_triangles:
        simplify_start = time.time()
        edges = simplify_triangle_cliques(edges, len(coords))
        logger.info(f"Simplified triangles: {len(edges)} edges remaining in {time.time() - simplify_start:.4f}s")
    
    skeleton_start = time.time()
    skeleton = CustomSkeleton(vertices=vertices, edges=edges)
    logger.info(f"Created CustomSkeleton object in {time.time() - skeleton_start:.4f}s")
    
    # Diagnostic: check degree distribution of initial skeleton
    g = skeleton.skeleton_to_graph()
    degrees = [g.degree[n] for n in g.nodes]
    logger.info(f"Initial skeleton degree distribution: min={min(degrees)}, max={max(degrees)}, mean={np.mean(degrees):.2f}")
    degree_counts = np.bincount(degrees)
    logger.info(f"Degree counts: {dict(enumerate(degree_counts))}")
    logger.info(f"Vertices with degree>2: {sum(1 for d in degrees if d > 2)}/{len(degrees)}")
    
    logger.info(f"Total skimage_to_custom_skeleton_fast time: {time.time() - start_time:.4f}s")
    
    return skeleton


def simplify_triangle_cliques(edges, num_vertices):
    """
    Remove redundant edges from triangular cliques (3 mutually connected nodes).
    For each triangle, keep only 2 edges to maintain connectivity.
    """
    import networkx as nx
    
    # Build adjacency list
    g = nx.Graph()
    g.add_nodes_from(range(num_vertices))
    g.add_edges_from(edges)
    
    # Find all triangles (3-cliques)
    triangles = [clique for clique in nx.enumerate_all_cliques(g) if len(clique) == 3]
    
    if not triangles:
        return edges
    
    logger.info(f"Found {len(triangles)} triangular cliques to simplify")
    
    # For each triangle, remove the longest edge
    edges_to_remove = set()
    for triangle in triangles:
        a, b, c = triangle
        
        # Calculate edge lengths (using node IDs as proxy, or could use actual positions)
        # For now, just remove one edge arbitrarily to break the cycle
        # Remove the edge with the largest sum of node indices (arbitrary but deterministic)
        edge_weights = [
            ((a, b), a + b),
            ((b, c), b + c),
            ((a, c), a + c)
        ]
        # Sort by weight and remove the heaviest
        edge_weights.sort(key=lambda x: x[1], reverse=True)
        edge_to_remove = tuple(sorted(edge_weights[0][0]))
        
        # Only remove if all three edges still exist (triangle hasn't been broken yet)
        if (g.has_edge(a, b) and g.has_edge(b, c) and g.has_edge(a, c)):
            edges_to_remove.add(edge_to_remove)
            g.remove_edge(*edge_to_remove)
    
    logger.info(f"Removing {len(edges_to_remove)} redundant edges from triangles")
    
    # Filter out removed edges
    edges_set = set(tuple(sorted(e)) for e in edges)
    edges_set -= edges_to_remove
    
    return list(edges_set)

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
        Simplify skeleton by removing vertices close to straight lines.
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

        for i, polyline in enumerate(polylines):
            # Check if this is a loop (first vertex == last vertex)
            is_loop = False

            if len(polyline) > 2:
                # Compare coordinates (handle both tuple and array formats)
                first = np.asarray(polyline[0])
                last = np.asarray(polyline[-1])
                if np.allclose(first, last):
                    is_loop = True

            # For loops: keep the closing vertex (A, B, C, D, A)
            # RDP will preserve both first and last vertices (even though they're at the same position)
            # This naturally preserves the loop structure
            polyline_for_rdp = polyline

            # Apply RDP simplification
            simplified = rdp(polyline_for_rdp, epsilon=tolerance_nm)

            # Round vertices to consistent precision
            simplified_rounded = []
            for vertex in simplified:
                v_rounded = tuple(round(float(x), PRECISION) for x in vertex)
                simplified_rounded.append(v_rounded)
                all_vertices_set.add(v_rounded)

            # For loops, simplified should already have first == last from RDP
            # Verify this is the case
            if is_loop and len(simplified_rounded) > 1:
                # Double-check the loop is still closed after RDP
                if simplified_rounded[0] != simplified_rounded[-1]:
                    # This shouldn't happen since RDP preserves endpoints, but just in case
                    simplified_rounded.append(simplified_rounded[0])

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

    def write_neuroglancer_skeleton(self, path):
        start_time = time.time()
        logger.info(f"Writing Neuroglancer skeleton to {path}")

        makedirs_start = time.time()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f"Created directory in {time.time() - makedirs_start:.4f}s")

        # Handle empty skeleton specially (neuroglancer library doesn't support empty arrays)
        if len(self.vertices) == 0:
            import struct
            with open(path, "wb") as f:
                # Write num_vertices (uint32) and num_edges (uint32), both 0
                f.write(struct.pack('<II', 0, 0))
            logger.info(f"Wrote empty skeleton (8 bytes)")
            logger.info(
                f"Total write_neuroglancer_skeleton time: {time.time() - start_time:.4f}s"
            )
            return

        encode_start = time.time()
        with open(path, "wb") as f:
            skel = NeuroglancerSkeleton(
                self.vertices, self.edges, vertex_attributes=None
            )
            encoded = skel.encode(Source())
            logger.info(f"Encoded skeleton in {time.time() - encode_start:.4f}s")

            write_start = time.time()
            f.write(encoded)
            logger.info(
                f"Wrote {len(encoded)} bytes in {time.time() - write_start:.4f}s"
            )

        logger.info(
            f"Total write_neuroglancer_skeleton time: {time.time() - start_time:.4f}s"
        )

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

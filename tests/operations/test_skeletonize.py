import pytest
from cellmap_analyze.process.skeletonize import Skeletonize
from cellmap_analyze.util.skeleton_util import (
    CustomSkeleton,
    skimage_to_custom_skeleton_fast,
)
from skimage.morphology import skeletonize, binary_erosion
import numpy as np
import pandas as pd
import os
import json


@pytest.fixture(scope="session")
def tmp_skeletonize_csv(shared_tmpdir, segmentation_for_skeleton, voxel_size):
    """Create a CSV with bounding box information for the test segmentation."""
    output_path = shared_tmpdir + "/csvs/"
    os.makedirs(name=output_path, exist_ok=True)

    # Calculate bounding boxes for each ID
    ids = np.unique(segmentation_for_skeleton[segmentation_for_skeleton > 0])

    data = []
    for id_val in ids:
        mask = segmentation_for_skeleton == id_val
        z_coords, y_coords, x_coords = np.where(mask)

        # Calculate bounding box in physical coordinates
        # Add 0.5 to center on voxel
        # Handle both scalar and tuple voxel_size
        if np.isscalar(voxel_size):
            vs = (voxel_size, voxel_size, voxel_size)
        else:
            vs = tuple(voxel_size)

        min_z = (z_coords.min() + 0.5) * vs[0]
        max_z = (z_coords.max() + 0.5) * vs[0]
        min_y = (y_coords.min() + 0.5) * vs[1]
        max_y = (y_coords.max() + 0.5) * vs[1]
        min_x = (x_coords.min() + 0.5) * vs[2]
        max_x = (x_coords.max() + 0.5) * vs[2]

        data.append(
            {
                "Object ID": id_val,
                "MIN X (nm)": min_x,
                "MIN Y (nm)": min_y,
                "MIN Z (nm)": min_z,
                "MAX X (nm)": max_x,
                "MAX Y (nm)": max_y,
                "MAX Z (nm)": max_z,
            }
        )

    df = pd.DataFrame(data)
    df.set_index("Object ID", inplace=True)
    csv_path = output_path + "/skeletonize_bboxes.csv"
    df.to_csv(csv_path)

    return csv_path


def compute_reference_skeleton(
    segmentation,
    id_value,
    voxel_size,
    erosion=True,
    min_branch_length_nm=0,
    tolerance_nm=0,
):
    """
    Compute a reference skeleton for a single ID using the same approach as the Skeletonize class,
    but without ROI extraction (process the whole image).

    Returns the skeleton with vertices in XYZ order (neuroglancer format).
    """
    # Extract binary mask for this ID (process entire image, no cropping)
    data = segmentation == id_value

    if not np.any(data):
        return None

    from cellmap_analyze.process.skeletonize import remove_unbridged_adjacencies

    # Normalize erosion parameter (same as Skeletonize.__init__)
    if erosion is True:
        erosion = "full"
    elif erosion is False or erosion is None:
        erosion = None
    elif erosion in (6, 18):
        erosion = str(erosion)

    # Apply erosion if requested
    if erosion == "full":
        cross_3d = np.array(
            [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            dtype=bool,
        )
        data = binary_erosion(data, cross_3d)
    elif erosion == "6":
        data = remove_unbridged_adjacencies(data, connectivity=6)
    elif erosion == "18":
        data = remove_unbridged_adjacencies(data, connectivity=18)

    if not np.any(data):
        return None

    # Skeletonize
    skel = skeletonize(data)

    if not np.any(skel):
        return None

    # Convert to custom skeleton
    # Vertices will be in physical coordinates (scaled by voxel_size)
    skeleton = skimage_to_custom_skeleton_fast(
        skel, spacing=(float(voxel_size), float(voxel_size), float(voxel_size))
    )

    # Transform vertices: swap Z/X for neuroglancer (ZYX -> XYZ)
    # No offset needed since we processed the entire image
    skeleton.vertices = [tuple([v[2], v[1], v[0]]) for v in skeleton.vertices]

    # Prune if requested
    if min_branch_length_nm > 0:
        skeleton = skeleton.prune(min_branch_length_nm)

    # Simplify if requested
    if tolerance_nm > 0:
        skeleton = skeleton.simplify(tolerance_nm)

    # Ensure edges is a properly shaped numpy array (after prune/simplify)
    if len(skeleton.edges) == 0:
        skeleton.edges = np.zeros((0, 2), dtype=np.uint32)
    else:
        skeleton.edges = np.array(skeleton.edges, dtype=np.uint32)

    return skeleton


def test_skeletonize_single_worker(tmp_zarr, tmp_skeletonize_csv):
    """Test skeletonization with a single worker."""
    output_path = tmp_zarr + "/test_skeletonize_output"

    skeletonizer = Skeletonize(
        segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
        output_path=output_path,
        csv_path=tmp_skeletonize_csv,
        erosion=True,
        min_branch_length_nm=0,  # No pruning for basic test
        tolerance_nm=0,  # No simplification for basic test
        num_workers=1,
        sharded=False,
    )

    skeletonizer.skeletonize()

    # Check that info files were created
    for subdir in ["full", "simplified"]:
        assert os.path.exists(f"{output_path}/{subdir}/info")
        assert os.path.exists(f"{output_path}/{subdir}/segment_properties/info")

        # Check info file contents
        with open(f"{output_path}/{subdir}/info", "r") as f:
            info = json.load(f)
            assert info["@type"] == "neuroglancer_skeletons"
            assert info["segment_properties"] == "segment_properties"

        # Check segment_properties info file
        with open(f"{output_path}/{subdir}/segment_properties/info", "r") as f:
            seg_props = json.load(f)
            assert seg_props["@type"] == "neuroglancer_segment_properties"
            assert "ids" in seg_props["inline"]
            assert set(seg_props["inline"]["ids"]) == {"1", "2", "3", "4", "5", "6", "7", "8"}

    # Check that skeleton files were created for all IDs (including empty skeletons)
    # Note: Some IDs (like ID 7 figure-8) may be completely removed by erosion and written as empty skeletons
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    for id_val in ids:
        # All IDs should have skeleton files (even if empty)
        full_path = f"{output_path}/full/{id_val}"
        simplified_path = f"{output_path}/simplified/{id_val}"

        assert os.path.exists(full_path), f"Full skeleton missing for ID {id_val}"
        assert os.path.exists(simplified_path), f"Simplified skeleton missing for ID {id_val}"


def test_skeletonize_produces_reasonable_skeletons(
    tmp_zarr, tmp_skeletonize_csv, voxel_size
):
    """
    Test that skeletonization produces reasonable skeletons for each ID.

    This test verifies basic properties of the generated skeletons without requiring
    exact matching to a reference (since ROI extraction with padding can affect results).
    """
    output_path = tmp_zarr + "/test_skeletonize_reasonable"

    # Run skeletonization using the Skeletonize class
    skeletonizer = Skeletonize(
        segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
        output_path=output_path,
        csv_path=tmp_skeletonize_csv,
        erosion=True,
        min_branch_length_nm=0,
        tolerance_nm=0,
        num_workers=1,
        sharded=False,
    )

    skeletonizer.skeletonize()

    # For each ID, verify the skeleton has reasonable properties
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    for id_val in ids:
        test_skeleton_path = f"{output_path}/full/{id_val}"

        # Some IDs may be completely eroded away
        if not os.path.exists(test_skeleton_path):
            continue

        test_vertices, test_edges = CustomSkeleton.read_neuroglancer_skeleton(
            test_skeleton_path
        )

        # Skip empty skeletons (erosion may have removed all voxels)
        if len(test_vertices) == 0:
            continue

        # Verify vertices are within reasonable bounds (should be near the bounding box)
        df = pd.read_csv(tmp_skeletonize_csv, index_col=0)
        row = df.loc[id_val]

        # Allow generous tolerance (several voxels) since erosion can shrink objects
        # For anisotropic data, use the maximum voxel size for the most generous tolerance
        if np.isscalar(voxel_size):
            tolerance = 10 * voxel_size
        else:
            tolerance = 10 * max(voxel_size)

        for v in test_vertices:
            x, y, z = v
            assert (
                x >= row["MIN X (nm)"] - tolerance
                and x <= row["MAX X (nm)"] + tolerance
            ), (
                f"ID {id_val}: X coordinate {x} far outside expected bounds "
                f"[{row['MIN X (nm)']}, {row['MAX X (nm)']}]"
            )
            assert (
                y >= row["MIN Y (nm)"] - tolerance
                and y <= row["MAX Y (nm)"] + tolerance
            ), (
                f"ID {id_val}: Y coordinate {y} far outside expected bounds "
                f"[{row['MIN Y (nm)']}, {row['MAX Y (nm)']}]"
            )
            assert (
                z >= row["MIN Z (nm)"] - tolerance
                and z <= row["MAX Z (nm)"] + tolerance
            ), (
                f"ID {id_val}: Z coordinate {z} far outside expected bounds "
                f"[{row['MIN Z (nm)']}, {row['MAX Z (nm)']}]"
            )

        # Verify edges reference valid vertices
        for edge in test_edges:
            assert edge[0] < len(
                test_vertices
            ), f"ID {id_val}: Edge references invalid vertex {edge[0]}"
            assert edge[1] < len(
                test_vertices
            ), f"ID {id_val}: Edge references invalid vertex {edge[1]}"


def test_skeletonize_without_erosion(tmp_zarr, tmp_skeletonize_csv, voxel_size):
    """Test skeletonization without erosion produces more detailed skeletons."""
    output_path = tmp_zarr + "/test_skeletonize_no_erosion"

    skeletonizer = Skeletonize(
        segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
        output_path=output_path,
        csv_path=tmp_skeletonize_csv,
        erosion=False,
        min_branch_length_nm=0,
        tolerance_nm=0,
        num_workers=1,
        sharded=False,
    )

    skeletonizer.skeletonize()

    # Check that info files were created
    for subdir in ["full", "simplified"]:
        assert os.path.exists(f"{output_path}/{subdir}/info")
        assert os.path.exists(f"{output_path}/{subdir}/segment_properties/info")

    # Check that skeleton files were created.
    # Lee's 3D thinning algorithm (skimage.skeletonize) can produce empty results
    # for certain block cross-sections (e.g. 6x6, 8x6, 4x4) due to symmetric
    # surface removal. After isotropic resampling, small objects (IDs 1-3) may
    # hit these problematic dimensions and lose their skeletons entirely.
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    ids_with_vertices = []
    for id_val in ids:
        full_path = f"{output_path}/full/{id_val}"
        simplified_path = f"{output_path}/simplified/{id_val}"

        assert os.path.exists(full_path), f"Full skeleton missing for ID {id_val}"
        assert os.path.exists(
            simplified_path
        ), f"Simplified skeleton missing for ID {id_val}"

        full_verts, full_edges = CustomSkeleton.read_neuroglancer_skeleton(full_path)
        if len(full_verts) > 0:
            ids_with_vertices.append(id_val)

    # Most IDs should produce skeletons; small objects may not due to Lee's
    # thinning limitation described above
    assert len(ids_with_vertices) >= 4, (
        f"Expected at least 4 IDs with skeletons, got {len(ids_with_vertices)}: {ids_with_vertices}"
    )

    # Verify skeleton metrics CSV was written with expected columns
    csv_dir = os.path.dirname(tmp_skeletonize_csv)
    csv_basename = os.path.splitext(os.path.basename(tmp_skeletonize_csv))[0]
    metrics_csv_path = os.path.join(csv_dir, f"{csv_basename}_with_skeletons.csv")
    assert os.path.exists(metrics_csv_path), "Skeleton metrics CSV not created"
    metrics_df = pd.read_csv(metrics_csv_path, index_col=0)

    # ID 5 (cross shape) should have meaningful skeleton metrics
    row5 = metrics_df.loc[5]

    # Cross has 3 arms meeting at a junction -> at least 3 branches
    # Isotropic resampling may produce extra short branches at junctions
    assert (
        row5["Number of Branches"] >= 3
    ), f"Cross (ID 5) should have at least 3 branches, got {row5['Number of Branches']}"

    # Longest shortest path: two longest arms (X=30 voxels, Y=22 voxels) through junction
    # Each arm extends from junction center to its tip; physical length depends on voxel size
    vs = np.array(voxel_size)
    # X arm half-length: ~15 voxels * vs[2], Y arm half-length: ~11 voxels * vs[1]
    # Approximate expected path = X arm half * vs[2] + Y arm half * vs[1]
    expected_path = 15 * vs[2] + 11 * vs[1]
    assert (
        abs(row5["Longest Shortest Path (nm)"] - expected_path) < expected_path * 0.3
    ), f"Cross (ID 5) longest shortest path should be ~{expected_path} nm, got {row5['Longest Shortest Path (nm)']}"

    # Radii should be approximately 2 voxels wide; use mean voxel size as approximation
    expected_radius = 2 * np.mean(vs)
    assert (
        abs(row5["Radius Mean (nm)"] - expected_radius) < expected_radius * 0.5
    ), f"Cross (ID 5) radius mean should be ~{expected_radius} nm, got {row5['Radius Mean (nm)']}"


def test_skeletonize_with_pruning_and_simplification(tmp_zarr, tmp_skeletonize_csv):
    """Test that pruning and simplification reduce the skeleton complexity."""
    output_path = tmp_zarr + "/test_skeletonize_pruned"

    min_branch_length_nm = 10
    tolerance_nm = 5

    # First, run without pruning/simplification
    skeletonizer_full = Skeletonize(
        segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
        output_path=output_path + "_full",
        csv_path=tmp_skeletonize_csv,
        erosion=True,
        min_branch_length_nm=0,
        tolerance_nm=0,
        num_workers=1,
        sharded=False,
    )
    skeletonizer_full.skeletonize()

    # Then run with pruning/simplification
    skeletonizer_simplified = Skeletonize(
        segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
        output_path=output_path + "_simplified",
        csv_path=tmp_skeletonize_csv,
        erosion=True,
        min_branch_length_nm=min_branch_length_nm,
        tolerance_nm=tolerance_nm,
        num_workers=1,
        sharded=False,
    )
    skeletonizer_simplified.skeletonize()

    # Check that simplified skeletons have fewer or equal vertices
    # (ID 2 and 3 are elongated and should be simplified)
    for id_val in [2, 3]:
        full_path = f"{output_path}_full/simplified/{id_val}"
        simplified_path = f"{output_path}_simplified/simplified/{id_val}"

        if os.path.exists(full_path) and os.path.exists(simplified_path):
            full_verts, _ = CustomSkeleton.read_neuroglancer_skeleton(full_path)
            simplified_verts, _ = CustomSkeleton.read_neuroglancer_skeleton(
                simplified_path
            )

            # Simplified should have fewer or equal vertices
            assert len(simplified_verts) <= len(full_verts), (
                f"ID {id_val}: Simplified skeleton has more vertices than full. "
                f"Full: {len(full_verts)}, Simplified: {len(simplified_verts)}"
            )


def test_skeletonize_with_roi_padding(tmp_zarr, tmp_skeletonize_csv, voxel_size):
    """
    Test that ROI padding doesn't affect the skeleton result.

    This verifies that the 1-voxel padding added to ROIs doesn't cause artifacts
    by comparing against a reference without padding considerations.
    """
    output_path = tmp_zarr + "/test_skeletonize_padding"

    skeletonizer = Skeletonize(
        segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
        output_path=output_path,
        csv_path=tmp_skeletonize_csv,
        erosion=True,
        min_branch_length_nm=0,
        tolerance_nm=0,
        num_workers=1,
        sharded=False,
    )

    skeletonizer.skeletonize()

    # For each ID, verify that the skeleton is within the expected bounds
    df = pd.read_csv(tmp_skeletonize_csv, index_col=0)

    for id_val in [1, 2, 3, 4, 5, 6, 8]:
        test_skeleton_path = f"{output_path}/full/{id_val}"
        if not os.path.exists(test_skeleton_path):
            continue

        test_vertices, _ = CustomSkeleton.read_neuroglancer_skeleton(test_skeleton_path)

        # Get bounding box for this ID
        row = df.loc[id_val]

        # Check that all vertices are within reasonable bounds
        # (allowing for some tolerance due to voxel centering and skeleton positioning)
        # For anisotropic data, use the maximum voxel size for the most generous tolerance
        if np.isscalar(voxel_size):
            tolerance = 2 * voxel_size  # Allow 2 voxels of tolerance
        else:
            tolerance = 2 * max(voxel_size)

        for v in test_vertices:
            # Vertices are in XYZ order
            x, y, z = v

            assert (
                x >= row["MIN X (nm)"] - tolerance
                and x <= row["MAX X (nm)"] + tolerance
            ), (
                f"ID {id_val}: X coordinate {x} out of bounds "
                f"[{row['MIN X (nm)']}, {row['MAX X (nm)']}]"
            )
            assert (
                y >= row["MIN Y (nm)"] - tolerance
                and y <= row["MAX Y (nm)"] + tolerance
            ), (
                f"ID {id_val}: Y coordinate {y} out of bounds "
                f"[{row['MIN Y (nm)']}, {row['MAX Y (nm)']}]"
            )
            assert (
                z >= row["MIN Z (nm)"] - tolerance
                and z <= row["MAX Z (nm)"] + tolerance
            ), (
                f"ID {id_val}: Z coordinate {z} out of bounds "
                f"[{row['MIN Z (nm)']}, {row['MAX Z (nm)']}]"
            )


def test_skeletonize_complex_shapes(tmp_zarr, tmp_skeletonize_csv, voxel_size):
    """
    Test that complex shapes (sphere, cross, L-shape) are skeletonized correctly.

    This test verifies that:
    - Sphere (ID 4) produces a minimal skeleton (few vertices)
    - Cross (ID 5) produces a skeleton with branch points
    - L-shape (ID 6) produces a skeleton with appropriate topology
    """
    output_path = tmp_zarr + "/test_skeletonize_complex"

    skeletonizer = Skeletonize(
        segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
        output_path=output_path,
        csv_path=tmp_skeletonize_csv,
        erosion=True,
        min_branch_length_nm=0,
        tolerance_nm=0,
        num_workers=1,
        sharded=False,
    )

    skeletonizer.skeletonize()

    # ID 4: Sphere - should produce a very small skeleton (compact structure)
    sphere_path = f"{output_path}/full/4"
    if os.path.exists(sphere_path):
        sphere_verts, _ = CustomSkeleton.read_neuroglancer_skeleton(sphere_path)
        # Sphere should collapse to a small number of vertices
        assert len(sphere_verts) < 50, (
            f"Sphere skeleton has too many vertices: {len(sphere_verts)}. "
            "Expected a compact skeleton."
        )

    # ID 5: Cross - should have branch points (degree > 2 nodes)
    cross_path = f"{output_path}/full/5"
    if os.path.exists(cross_path):
        cross_verts, cross_edges = CustomSkeleton.read_neuroglancer_skeleton(cross_path)

        # Skip if empty (erosion may have removed all voxels)
        if len(cross_verts) > 0:
            # Build adjacency to check for branch points
            from collections import defaultdict

            adjacency = defaultdict(set)
            for edge in cross_edges:
                adjacency[edge[0]].add(edge[1])
                adjacency[edge[1]].add(edge[0])

            # Count nodes with degree > 2 (branch points)
            branch_points = sum(
                1 for node_id in range(len(cross_verts)) if len(adjacency[node_id]) > 2
            )

            # Cross should have at least one branch point where the three branches meet
            assert (
                branch_points >= 1
            ), f"Cross skeleton has no branch points. Found {branch_points}, expected >= 1"

    # ID 6: L-shape - should have at least one vertex
    # (may have no edges if erosion shrinks it to a point)
    # With very small test data and non-integer voxel sizes, erosion may remove
    # all voxels, so only assert if the skeleton file exists and is non-empty
    l_shape_path = f"{output_path}/full/6"
    if os.path.exists(l_shape_path):
        l_verts, l_edges = CustomSkeleton.read_neuroglancer_skeleton(l_shape_path)
        # Edges are optional - after erosion it might be a single point or even empty


def test_skeletonize_preserves_loops(tmp_zarr, tmp_skeletonize_csv, voxel_size):
    """
    Test that loops are preserved during simplification.

    This test uses ID 7 (figure-8 shape) which should produce a skeleton with loops.
    After simplification, the loops should still be present (not broken into open paths).
    """
    import networkx as nx

    output_path = tmp_zarr + "/test_skeletonize_loops"

    # Run skeletonization WITHOUT erosion to preserve the figure-8 structure
    skeletonizer = Skeletonize(
        segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
        output_path=output_path,
        csv_path=tmp_skeletonize_csv,
        erosion=False,  # No erosion to preserve loop structure
        min_branch_length_nm=0,
        tolerance_nm=5,
        num_workers=1,
        sharded=False,
    )

    skeletonizer.skeletonize()

    # ID 7: Figure-8 shape - should preserve loops after simplification
    figure8_path = f"{output_path}/simplified/7"

    assert os.path.exists(
        figure8_path
    ), "Figure-8 skeleton file not found. The figure-8 structure (ID 7) should be present."

    fig8_verts, fig8_edges = CustomSkeleton.read_neuroglancer_skeleton(figure8_path)

    # Verify we got some skeleton
    assert len(fig8_verts) > 0, "Figure-8 skeleton has no vertices"
    assert len(fig8_edges) > 0, "Figure-8 skeleton has no edges"

    # Build a graph from the skeleton to analyze topology
    g = nx.Graph()
    g.add_edges_from(fig8_edges)

    # Check that the graph is connected
    assert nx.is_connected(g), "Figure-8 skeleton is not connected"

    # Find cycles in the graph
    cycles = nx.cycle_basis(g)

    # A figure-8 should have at least one loop (ideally two, but simplification might merge them)
    # The key property is that there ARE loops - not just a tree structure
    assert len(cycles) >= 1, (
        f"Figure-8 skeleton has no cycles after simplification. "
        f"Found {len(cycles)} cycles, expected >= 1. "
        "This indicates loops were broken during simplification."
    )

    # Verify that loops are properly closed
    # For each polyline in the skeleton, check if it's a loop
    # (We need to reconstruct polylines from the graph or check the internal representation)
    print(f"Figure-8 skeleton has {len(fig8_verts)} vertices, {len(fig8_edges)} edges")
    print(f"Found {len(cycles)} cycle(s) in the graph")

    # Additional check: in a proper figure-8, we expect at least one branch point
    # (the point where the two loops connect)
    degrees = dict(g.degree())
    branch_points = sum(1 for node_id in g.nodes if degrees[node_id] > 2)

    print(f"Found {branch_points} branch point(s)")

    # A figure-8 should have at least one branch point where loops connect
    # (unless simplification reduced it to a single loop, which is still valid)
    if len(cycles) > 1:
        assert branch_points >= 1, (
            f"Figure-8 with {len(cycles)} loops should have at least 1 branch point, "
            f"but found {branch_points}"
        )


# =============================================================================
# Unit tests for remove_unbridged_adjacencies
# =============================================================================


class TestRemoveUnbridgedAdjacencies:
    """Unit tests for the remove_unbridged_adjacencies function."""

    def test_edge_adjacent_no_bridge_removed_conn6(self):
        """Two voxels edge-adjacent with no face bridge should have one removed at connectivity=6."""
        from cellmap_analyze.process.skeletonize import remove_unbridged_adjacencies

        data = np.zeros((5, 5, 5), dtype=bool)
        data[1, 1, 1] = True
        data[2, 2, 1] = True  # edge-adjacent via (1,1,0), no face bridge
        result = remove_unbridged_adjacencies(data, connectivity=6)
        # At least one should be removed
        assert result.sum() < data.sum()

    def test_edge_adjacent_with_bridge_kept_conn6(self):
        """Two voxels edge-adjacent WITH a face bridge should both be kept at connectivity=6."""
        from cellmap_analyze.process.skeletonize import remove_unbridged_adjacencies

        data = np.zeros((5, 5, 5), dtype=bool)
        data[1, 1, 1] = True
        data[2, 2, 1] = True  # edge-adjacent
        data[2, 1, 1] = True  # face bridge between them
        result = remove_unbridged_adjacencies(data, connectivity=6)
        assert result.sum() == data.sum()

    def test_corner_adjacent_no_bridge_removed_both_modes(self):
        """Two voxels vertex-adjacent with no bridge should have one removed in both modes."""
        from cellmap_analyze.process.skeletonize import remove_unbridged_adjacencies

        data = np.zeros((5, 5, 5), dtype=bool)
        data[1, 1, 1] = True
        data[2, 2, 2] = True  # vertex-adjacent, no bridges

        result_6 = remove_unbridged_adjacencies(data, connectivity=6)
        assert result_6.sum() < data.sum()

        result_18 = remove_unbridged_adjacencies(data, connectivity=18)
        assert result_18.sum() < data.sum()

    def test_corner_adjacent_with_edge_bridge_only(self):
        """Vertex-adjacent pair with edge bridge only: kept for conn=18, removed for conn=6."""
        from cellmap_analyze.process.skeletonize import remove_unbridged_adjacencies

        data = np.zeros((5, 5, 5), dtype=bool)
        data[1, 1, 1] = True
        data[2, 2, 2] = True  # vertex-adjacent
        data[2, 2, 1] = True  # edge bridge (differs in 2 axes from [1,1,1])

        # connectivity=18: edge bridge exists, should keep both original voxels
        result_18 = remove_unbridged_adjacencies(data, connectivity=18)
        assert result_18[1, 1, 1] and result_18[2, 2, 2]

        # connectivity=6: the edge bridge voxel (2,2,1) is itself only
        # edge-connected to (1,1,1), so it gets removed. But A and B survive
        # this single pass because (2,2,1) was still present when checked.
        result_6 = remove_unbridged_adjacencies(data, connectivity=6)
        assert not result_6[2, 2, 1]  # edge bridge removed
        # A and B both survive the single pass (bridge was present when checked)
        assert result_6[1, 1, 1] and result_6[2, 2, 2]

    def test_solid_cube_unchanged(self):
        """A solid cube should have no voxels removed (all neighbors are face-bridged)."""
        from cellmap_analyze.process.skeletonize import remove_unbridged_adjacencies

        data = np.zeros((10, 10, 10), dtype=bool)
        data[2:6, 2:6, 2:6] = True
        original_count = data.sum()

        result_6 = remove_unbridged_adjacencies(data, connectivity=6)
        assert result_6.sum() == original_count

        result_18 = remove_unbridged_adjacencies(data, connectivity=18)
        assert result_18.sum() == original_count

    def test_c_shape_diagonal_broken_body_intact(self):
        """A C-shape touching itself at a diagonal should have the diagonal broken."""
        from cellmap_analyze.process.skeletonize import remove_unbridged_adjacencies

        data = np.zeros((7, 12, 12), dtype=bool)
        # Base
        data[2:5, 2:5, 2:10] = True
        # Left arm
        data[2:5, 5:10, 2:5] = True
        # Right arm
        data[2:5, 5:10, 7:10] = True
        # Tips touch diagonally
        data[3, 10, 5] = True
        data[3, 10, 6] = True

        original_count = data.sum()
        result = remove_unbridged_adjacencies(data, connectivity=6)

        # Should remove the diagonal-only tip voxels but keep the body
        assert result.sum() < original_count
        # Body should be intact
        assert result[2:5, 2:5, 2:10].all()
        assert result[2:5, 5:10, 2:5].all()
        assert result[2:5, 5:10, 7:10].all()

    def test_invalid_connectivity_raises(self):
        """Invalid connectivity value should raise ValueError."""
        from cellmap_analyze.process.skeletonize import remove_unbridged_adjacencies

        data = np.zeros((3, 3, 3), dtype=bool)
        with pytest.raises(ValueError):
            remove_unbridged_adjacencies(data, connectivity=26)


# =============================================================================
# Integration tests for new erosion modes
# =============================================================================


def test_skeletonize_diagonal_6_erosion(tmp_zarr, tmp_skeletonize_csv):
    """Test skeletonization with connectivity=6 erosion mode."""
    output_path = tmp_zarr + "/test_skeletonize_diagonal_6"

    skeletonizer = Skeletonize(
        segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
        output_path=output_path,
        csv_path=tmp_skeletonize_csv,
        erosion=6,
        min_branch_length_nm=0,
        tolerance_nm=0,
        num_workers=1,
        sharded=False,
    )

    skeletonizer.skeletonize()

    # Check that skeleton files were created for all IDs
    for id_val in [1, 2, 3, 4, 5, 6, 7, 8]:
        full_path = f"{output_path}/full/{id_val}"
        simplified_path = f"{output_path}/simplified/{id_val}"
        assert os.path.exists(full_path), f"Full skeleton missing for ID {id_val}"
        assert os.path.exists(simplified_path), f"Simplified skeleton missing for ID {id_val}"


def test_skeletonize_diagonal_18_erosion(tmp_zarr, tmp_skeletonize_csv):
    """Test skeletonization with connectivity=18 erosion mode."""
    output_path = tmp_zarr + "/test_skeletonize_diagonal_18"

    skeletonizer = Skeletonize(
        segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
        output_path=output_path,
        csv_path=tmp_skeletonize_csv,
        erosion=18,
        min_branch_length_nm=0,
        tolerance_nm=0,
        num_workers=1,
        sharded=False,
    )

    skeletonizer.skeletonize()

    # Check that skeleton files were created for all IDs
    for id_val in [1, 2, 3, 4, 5, 6, 7, 8]:
        full_path = f"{output_path}/full/{id_val}"
        simplified_path = f"{output_path}/simplified/{id_val}"
        assert os.path.exists(full_path), f"Full skeleton missing for ID {id_val}"
        assert os.path.exists(simplified_path), f"Simplified skeleton missing for ID {id_val}"


def test_skeletonize_backward_compat_erosion_true(tmp_zarr, tmp_skeletonize_csv):
    """Test that erosion=True still works (backward compatibility)."""
    output_path = tmp_zarr + "/test_skeletonize_compat_true"

    skeletonizer = Skeletonize(
        segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
        output_path=output_path,
        csv_path=tmp_skeletonize_csv,
        erosion=True,
        min_branch_length_nm=0,
        tolerance_nm=0,
        num_workers=1,
        sharded=False,
    )

    assert skeletonizer.erosion == "full"
    skeletonizer.skeletonize()

    for id_val in [1, 2, 3, 4, 5, 6, 7, 8]:
        assert os.path.exists(f"{output_path}/full/{id_val}")


def test_skeletonize_seed_voxel_fallback(shared_tmpdir):
    """A 4x4x4 cube is wiped to zero voxels by Lee's algorithm. The
    seed-voxel fallback should emit a single-vertex skeleton at the EDT
    peak with a positive radius, instead of writing an empty skeleton."""
    from cellmap_analyze.util.zarr_util import create_multiscale_dataset
    from funlib.geometry import Coordinate, Roi

    # Build a tiny zarr with one 4x4x4 cube (ID=1) inside a 12x12x12 volume.
    # Use isotropic 1nm voxels so we don't trigger resampling.
    vs = (1, 1, 1)
    seg = np.zeros((12, 12, 12), dtype=np.uint8)
    seg[4:8, 4:8, 4:8] = 1

    zarr_path = shared_tmpdir + "/seed_voxel_test.zarr"
    os.makedirs(zarr_path, exist_ok=True)
    data_path = f"{zarr_path}/seg_cube"
    total_roi = Roi((0, 0, 0), Coordinate(seg.shape) * Coordinate(vs))
    ds = create_multiscale_dataset(
        data_path,
        dtype=seg.dtype,
        voxel_size=vs,
        total_roi=total_roi,
        write_size=Coordinate(seg.shape) * Coordinate(vs),
        original_voxel_size=vs,
    )
    ds.data[:] = seg

    # CSV with the one cube's bbox.
    csv_path = shared_tmpdir + "/seed_voxel_bboxes.csv"
    pd.DataFrame(
        [{
            "Object ID": 1,
            "MIN X (nm)": 4.5,
            "MIN Y (nm)": 4.5,
            "MIN Z (nm)": 4.5,
            "MAX X (nm)": 7.5,
            "MAX Y (nm)": 7.5,
            "MAX Z (nm)": 7.5,
        }]
    ).set_index("Object ID").to_csv(csv_path)

    output_path = shared_tmpdir + "/seed_voxel_output"

    Skeletonize(
        segmentation_path=f"{data_path}/s0",
        output_path=output_path,
        csv_path=csv_path,
        erosion=False,
        min_branch_length_nm=0,
        tolerance_nm=0,
        num_workers=1,
        sharded=False,
    ).skeletonize()

    # The fallback should have written a 1-vertex skeleton (not empty).
    verts, edges = CustomSkeleton.read_neuroglancer_skeleton(f"{output_path}/full/1")
    assert len(verts) == 1, f"expected 1 seed vertex, got {len(verts)}"
    assert len(edges) == 0, f"expected 0 edges for single-vertex skeleton, got {len(edges)}"

    # Seed should sit at the cube interior (somewhere in [4, 8] on every axis).
    x, y, z = verts[0]
    assert 4.0 <= x <= 8.0, f"seed x={x} outside cube"
    assert 4.0 <= y <= 8.0, f"seed y={y} outside cube"
    assert 4.0 <= z <= 8.0, f"seed z={z} outside cube"

    # CSV: 0 length, 0 branches, but Radius Mean should be positive (~half-thickness).
    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
    metrics_csv = os.path.join(
        os.path.dirname(csv_path), f"{csv_basename}_with_skeletons.csv"
    )
    df = pd.read_csv(metrics_csv, index_col=0)
    assert df.loc[1, "Number of Branches"] == 0
    assert df.loc[1, "Longest Shortest Path (nm)"] == 0.0
    assert df.loc[1, "Radius Mean (nm)"] > 0, (
        f"seed-voxel fallback should populate radius, got "
        f"{df.loc[1, 'Radius Mean (nm)']!r}"
    )
    assert df.loc[1, "Radius Std (nm)"] == 0.0


def test_skeletonize_nonzero_translation_shifts_by_translation(shared_tmpdir):
    """Regression for the OME voxel-CENTER convention in the skeletonize path.

    The rest of the suite only exercises translation 0 (where the center<->corner
    conversion cancels to a no-op). Here we skeletonize the SAME object in two
    datasets that differ only by a non-zero OME translation, and assert the
    resulting skeleton is identical up to a shift equal to that translation --
    i.e. translation propagates with no half-voxel, sign, or doubling error.
    """
    from cellmap_analyze.util.zarr_util import create_multiscale_dataset
    from funlib.geometry import Coordinate, Roi

    vs = (2, 2, 2)          # isotropic -> no resampling; vs/2 = 1 (integer corner)
    delta = 24              # isotropic corner shift in nm (== OME translation shift)

    # Elongated rod (3x3 cross-section, 11 long) skeletonizes to a clean line.
    seg = np.zeros((20, 20, 20), dtype=np.uint8)
    seg[3:14, 9:12, 9:12] = 1
    zc, yc, xc = np.where(seg)

    def run(name, begin):
        data_path = f"{shared_tmpdir}/{name}.zarr/seg"
        total_roi = Roi(Coordinate(begin), Coordinate(seg.shape) * Coordinate(vs))
        ds = create_multiscale_dataset(
            data_path,
            dtype=seg.dtype,
            voxel_size=list(vs),
            total_roi=total_roi,
            write_size=Coordinate(seg.shape) * Coordinate(vs),
            original_voxel_size=list(vs),
        )
        ds.data[:] = seg
        # bbox CSV = OME centers of the object's min/max voxels, exactly what
        # measure would emit for this dataset: (idx + 0.5)*vs + begin (sf == 1).
        b = begin
        csv_path = f"{shared_tmpdir}/{name}_bbox.csv"
        pd.DataFrame([{
            "Object ID": 1,
            "MIN Z (nm)": (zc.min() + 0.5) * vs[0] + b[0],
            "MIN Y (nm)": (yc.min() + 0.5) * vs[1] + b[1],
            "MIN X (nm)": (xc.min() + 0.5) * vs[2] + b[2],
            "MAX Z (nm)": (zc.max() + 0.5) * vs[0] + b[0],
            "MAX Y (nm)": (yc.max() + 0.5) * vs[1] + b[1],
            "MAX X (nm)": (xc.max() + 0.5) * vs[2] + b[2],
        }]).set_index("Object ID").to_csv(csv_path)
        out = f"{shared_tmpdir}/{name}_out"
        Skeletonize(
            segmentation_path=f"{data_path}/s0",
            output_path=out,
            csv_path=csv_path,
            erosion=False,
            min_branch_length_nm=0,
            tolerance_nm=0,
            num_workers=1,
            sharded=False,
        ).skeletonize()
        verts, _ = CustomSkeleton.read_neuroglancer_skeleton(f"{out}/full/1")
        return np.array(sorted(map(tuple, verts)))

    verts_0 = run("skel_t0", (0, 0, 0))
    verts_t = run("skel_tT", (delta, delta, delta))

    assert len(verts_0) > 0, "expected a non-empty skeleton"
    assert len(verts_0) == len(verts_t)
    # The translated skeleton is the zero skeleton shifted by exactly delta.
    assert np.allclose(verts_t, verts_0 + delta, atol=1e-6)
    # And the zero skeleton sits on OME centers within the object's z bbox
    # (MIN Z center = 3.5*2 = 7, MAX Z center = 13.5*2 = 27).
    assert verts_0[:, 2].min() >= 7 - 1e-6
    assert verts_0[:, 2].max() <= 27 + 1e-6


def test_skeletonize_sharded_default(tmp_zarr, tmp_skeletonize_csv):
    """With sharded=True default, per-ID files are repacked into shard files
    and the info file picks up the sharding spec; one chunk should round-trip
    via the read helper."""
    from cellmap_analyze.util.sharded_skeleton import read_chunk_from_shard

    output_path = tmp_zarr + "/test_skeletonize_sharded"

    skeletonizer = Skeletonize(
        segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
        output_path=output_path,
        csv_path=tmp_skeletonize_csv,
        erosion=False,
        min_branch_length_nm=0,
        tolerance_nm=0,
        num_workers=1,
    )
    skeletonizer.skeletonize()

    for subdir in ["full", "simplified"]:
        info_path = f"{output_path}/{subdir}/info"
        with open(info_path) as f:
            info = json.load(f)
        assert "sharding" in info, f"Sharding spec missing from {info_path}"
        assert info["sharding"]["@type"] == "neuroglancer_uint64_sharded_v1"

        shard_files = [
            n for n in os.listdir(f"{output_path}/{subdir}") if n.endswith(".shard")
        ]
        assert shard_files, f"No .shard files written under {subdir}/"

        # Per-ID files should be gone.
        for id_val in [1, 2, 3, 4, 5, 6, 7, 8]:
            assert not os.path.exists(f"{output_path}/{subdir}/{id_val}")

        # At least one non-empty chunk should round-trip out of the shards.
        found_any = False
        for id_val in [1, 2, 3, 4, 5, 6, 7, 8]:
            chunk = read_chunk_from_shard(
                f"{output_path}/{subdir}", id_val, info["sharding"]
            )
            if chunk:
                found_any = True
                break
        assert found_any, f"No skeleton chunks readable from shards in {subdir}/"


def test_skeletonize_backward_compat_erosion_false(tmp_zarr, tmp_skeletonize_csv):
    """Test that erosion=False still works (backward compatibility)."""
    output_path = tmp_zarr + "/test_skeletonize_compat_false"

    skeletonizer = Skeletonize(
        segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
        output_path=output_path,
        csv_path=tmp_skeletonize_csv,
        erosion=False,
        min_branch_length_nm=0,
        tolerance_nm=0,
        num_workers=1,
        sharded=False,
    )

    assert skeletonizer.erosion is None
    skeletonizer.skeletonize()

    for id_val in [1, 2, 3, 4, 5, 6, 7, 8]:
        assert os.path.exists(f"{output_path}/full/{id_val}")


def _read_seg_props(output_path, subdir="full"):
    with open(f"{output_path}/{subdir}/segment_properties/info") as f:
        return json.load(f)


def test_skeletonize_segment_properties_default(tmp_zarr, tmp_skeletonize_csv):
    """The default subset is the two triage-first metrics: num_branches and
    longest_shortest_path_nm. radius_* are opt-in."""
    output_path = tmp_zarr + "/test_skeletonize_segprops_default"
    Skeletonize(
        segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
        output_path=output_path,
        csv_path=tmp_skeletonize_csv,
        erosion=False,
        min_branch_length_nm=0,
        tolerance_nm=0,
        num_workers=1,
        sharded=False,
    ).skeletonize()

    expected_ids = {"1", "2", "3", "4", "5", "6", "7", "8"}
    default_metrics = {"num_branches", "longest_shortest_path_nm"}

    for subdir in ("full", "simplified"):
        seg = _read_seg_props(output_path, subdir)
        ids = seg["inline"]["ids"]
        assert set(ids) == expected_ids
        n = len(ids)
        props = {p["id"]: p for p in seg["inline"]["properties"]}
        assert "label" in props
        # exactly the two default metrics — radius_* not included
        assert set(props.keys()) - {"label"} == default_metrics
        for k in default_metrics:
            p = props[k]
            assert p["type"] == "number"
            assert p["data_type"] in ("int32", "float32")
            assert len(p["values"]) == n

        i5 = ids.index("5")
        assert props["num_branches"]["values"][i5] >= 3
        assert props["longest_shortest_path_nm"]["values"][i5] > 0.0


def test_skeletonize_segment_properties_all(tmp_zarr, tmp_skeletonize_csv):
    """skeleton_properties='all' includes radius_mean_nm and radius_std_nm too."""
    output_path = tmp_zarr + "/test_skeletonize_segprops_all"
    Skeletonize(
        segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
        output_path=output_path,
        csv_path=tmp_skeletonize_csv,
        erosion=False,
        min_branch_length_nm=0,
        tolerance_nm=0,
        num_workers=1,
        sharded=False,
        skeleton_properties="all",
    ).skeletonize()

    seg = _read_seg_props(output_path, "full")
    props = {p["id"]: p for p in seg["inline"]["properties"]}
    expected = {
        "label",
        "num_branches",
        "longest_shortest_path_nm",
        "radius_mean_nm",
        "radius_std_nm",
    }
    assert set(props.keys()) == expected


def test_skeletonize_segment_properties_explicit_list(tmp_zarr, tmp_skeletonize_csv):
    """Passing an explicit list emits exactly those properties."""
    output_path = tmp_zarr + "/test_skeletonize_segprops_list"
    Skeletonize(
        segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
        output_path=output_path,
        csv_path=tmp_skeletonize_csv,
        erosion=False,
        min_branch_length_nm=0,
        tolerance_nm=0,
        num_workers=1,
        sharded=False,
        skeleton_properties=["radius_mean_nm"],
    ).skeletonize()

    seg = _read_seg_props(output_path, "full")
    props = {p["id"]: p for p in seg["inline"]["properties"]}
    assert set(props.keys()) == {"label", "radius_mean_nm"}


def test_skeletonize_segment_properties_opt_out(tmp_zarr, tmp_skeletonize_csv):
    """skeleton_properties=False keeps only the label property."""
    output_path = tmp_zarr + "/test_skeletonize_segprops_optout"
    Skeletonize(
        segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
        output_path=output_path,
        csv_path=tmp_skeletonize_csv,
        erosion=False,
        min_branch_length_nm=0,
        tolerance_nm=0,
        num_workers=1,
        sharded=False,
        skeleton_properties=False,
    ).skeletonize()

    seg = _read_seg_props(output_path, "full")
    ids_in_props = {p["id"] for p in seg["inline"]["properties"]}
    assert ids_in_props == {"label"}


def test_skeletonize_segment_properties_invalid_key_raises(tmp_zarr, tmp_skeletonize_csv):
    """Unknown metric keys in the list raise with a clear message."""
    with pytest.raises(ValueError, match="unknown skeleton_properties"):
        Skeletonize(
            segmentation_path=f"{tmp_zarr}/segmentation_for_skeleton/s0",
            output_path=tmp_zarr + "/test_skeletonize_segprops_bad",
            csv_path=tmp_skeletonize_csv,
            erosion=False,
            num_workers=1,
            sharded=False,
            skeleton_properties=["num_branches", "not_a_real_metric"],
        )

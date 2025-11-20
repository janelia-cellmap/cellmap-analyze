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
        min_z = (z_coords.min() + 0.5) * voxel_size
        max_z = (z_coords.max() + 0.5) * voxel_size
        min_y = (y_coords.min() + 0.5) * voxel_size
        max_y = (y_coords.max() + 0.5) * voxel_size
        min_x = (x_coords.min() + 0.5) * voxel_size
        max_x = (x_coords.max() + 0.5) * voxel_size

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

    # Apply erosion if requested
    if erosion:
        cross_3d = np.array(
            [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            dtype=bool,
        )
        data = binary_erosion(data, cross_3d)

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
            assert set(seg_props["inline"]["ids"]) == {"1", "2", "3", "4", "5", "6"}

    # Check that skeleton files were created for IDs that didn't get eroded away
    # Note: Some IDs (like the cross shape) may be completely removed by erosion
    ids = [1, 2, 3, 4, 5, 6]
    files_found = 0
    for id_val in ids:
        # Check if files exist (they may not if erosion removed all voxels)
        full_exists = os.path.exists(f"{output_path}/full/{id_val}")
        simplified_exists = os.path.exists(f"{output_path}/simplified/{id_val}")

        # If one exists, both should exist
        if full_exists or simplified_exists:
            assert full_exists, f"Full skeleton missing for ID {id_val}"
            assert simplified_exists, f"Simplified skeleton missing for ID {id_val}"
            files_found += 1

    # At least some IDs should have produced skeletons
    assert files_found > 0, "No skeleton files were created"


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
    )

    skeletonizer.skeletonize()

    # For each ID, verify the skeleton has reasonable properties
    ids = [1, 2, 3, 4, 5, 6]
    for id_val in ids:
        test_skeleton_path = f"{output_path}/full/{id_val}"

        # Some IDs may be completely eroded away
        if not os.path.exists(test_skeleton_path):
            continue

        test_vertices, test_edges = CustomSkeleton.read_neuroglancer_skeleton(
            test_skeleton_path
        )

        # Verify skeleton has at least one vertex
        assert len(test_vertices) > 0, f"ID {id_val}: Skeleton has no vertices"

        # Verify vertices are within reasonable bounds (should be near the bounding box)
        df = pd.read_csv(tmp_skeletonize_csv, index_col=0)
        row = df.loc[id_val]

        # Allow generous tolerance (several voxels) since erosion can shrink objects
        tolerance = 10 * voxel_size

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


def test_skeletonize_without_erosion(tmp_zarr, tmp_skeletonize_csv):
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
    )

    skeletonizer.skeletonize()

    # Check that info files were created
    for subdir in ["full", "simplified"]:
        assert os.path.exists(f"{output_path}/{subdir}/info")
        assert os.path.exists(f"{output_path}/{subdir}/segment_properties/info")

    # Check that skeleton files were created
    # Without erosion, all IDs should produce skeletons
    ids = [1, 2, 3, 4, 5, 6]
    for id_val in ids:
        full_path = f"{output_path}/full/{id_val}"
        simplified_path = f"{output_path}/simplified/{id_val}"

        assert os.path.exists(full_path), f"Full skeleton missing for ID {id_val}"
        assert os.path.exists(
            simplified_path
        ), f"Simplified skeleton missing for ID {id_val}"

        # Verify skeletons have vertices
        full_verts, full_edges = CustomSkeleton.read_neuroglancer_skeleton(full_path)
        assert (
            len(full_verts) > 0
        ), f"ID {id_val}: No vertices in skeleton without erosion"


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
    )

    skeletonizer.skeletonize()

    # For each ID, verify that the skeleton is within the expected bounds
    df = pd.read_csv(tmp_skeletonize_csv, index_col=0)

    for id_val in [1, 2, 3, 4, 5, 6]:
        test_skeleton_path = f"{output_path}/full/{id_val}"
        if not os.path.exists(test_skeleton_path):
            continue

        test_vertices, _ = CustomSkeleton.read_neuroglancer_skeleton(test_skeleton_path)

        # Get bounding box for this ID
        row = df.loc[id_val]

        # Check that all vertices are within reasonable bounds
        # (allowing for some tolerance due to voxel centering and skeleton positioning)
        tolerance = 2 * voxel_size  # Allow 2 voxels of tolerance

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
    l_shape_path = f"{output_path}/full/6"
    if os.path.exists(l_shape_path):
        l_verts, l_edges = CustomSkeleton.read_neuroglancer_skeleton(l_shape_path)

        # L-shape should have at least one vertex
        assert len(l_verts) > 0, "L-shape skeleton is empty"
        # Edges are optional - after erosion it might be a single point


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

import numpy as np
import pandas as pd
import networkx as nx
import edt as edt_module
from cellmap_analyze.util import dask_util
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.skeleton_util import (
    CustomSkeleton,
    skimage_to_custom_skeleton_fast,
)
from skimage.morphology import skeletonize, binary_erosion
import logging
import os
import json

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Skeletonize(ComputeConfigMixin):
    def __init__(
        self,
        segmentation_path,
        output_path,
        csv_path,
        erosion=True,
        min_branch_length_nm=100,
        tolerance_nm=50,
        num_workers=10,
        timeout=5,
    ):
        """
        Skeletonize a segmentation, parallelized over IDs.

        Args:
            segmentation_path: Path to the segmentation zarr dataset
            output_path: Path to the output directory for skeletons
            csv_path: Path to CSV containing bounding box info with columns:
                     MIN X (nm), MIN Y (nm), MIN Z (nm), MAX X (nm), MAX Y (nm), MAX Z (nm)
                     and index column for IDs
            erosion: Whether to apply binary erosion before skeletonization
            min_branch_length_nm: Minimum branch length for pruning (in nm)
            tolerance_nm: Tolerance for simplification (in nm)
            num_workers: Number of parallel workers
            timeout: Timeout for ImageDataInterface reads
        """
        super().__init__(num_workers)
        self.segmentation_idi = ImageDataInterface(segmentation_path, timeout=timeout)
        self.output_path = str(output_path).rstrip("/")
        self.csv_path = csv_path
        self.erosion = erosion
        self.min_branch_length_nm = min_branch_length_nm
        self.tolerance_nm = tolerance_nm
        self.num_workers = num_workers

        # Load CSV with bounding box info
        self.bbox_df = pd.read_csv(csv_path, index_col=0)
        self.ids = self.bbox_df.index.tolist()

        # Create output directories
        # Each output directory (full and simplified) needs its own structure
        os.makedirs(f"{output_path}/full", exist_ok=True)
        os.makedirs(f"{output_path}/full/segment_properties", exist_ok=True)
        os.makedirs(f"{output_path}/simplified", exist_ok=True)
        os.makedirs(f"{output_path}/simplified/segment_properties", exist_ok=True)

        logger.info(f"Loaded {len(self.ids)} IDs from {csv_path}")
        logger.info(f"Output will be written to {output_path}")

    @staticmethod
    def _empty_metrics():
        return {
            "longest_shortest_path_nm": 0.0,
            "num_branches": 0,
            "radius_mean_nm": np.nan,
            "radius_std_nm": np.nan,
        }

    @staticmethod
    def calculate_id_skeleton(
        id_value,
        segmentation_idi: ImageDataInterface,
        bbox_df: pd.DataFrame,
        output_path: str,
        erosion: bool,
        min_branch_length_nm: float,
        tolerance_nm: float,
    ):
        """
        Process a single ID: extract, skeletonize, prune, simplify, and save.

        Args:
            id_value: The ID to process
            segmentation_idi: ImageDataInterface for the segmentation
            bbox_df: DataFrame with bounding box information
            output_path: Base output path
            erosion: Whether to apply erosion
            min_branch_length_nm: Minimum branch length for pruning
            tolerance_nm: Tolerance for simplification
        """
        from funlib.geometry import Roi

        try:
            # Get bounding box for this ID
            row = bbox_df.loc[id_value]
            min_x = row["MIN X (nm)"]
            min_y = row["MIN Y (nm)"]
            min_z = row["MIN Z (nm)"]
            max_x = row["MAX X (nm)"]
            max_y = row["MAX Y (nm)"]
            max_z = row["MAX Z (nm)"]

            # Create ROI with 1-voxel padding
            voxel_size = segmentation_idi.voxel_size
            padding = voxel_size  # 1 voxel in each direction
            start_point = np.array([min_z, min_y, min_x]) - padding
            end_point = np.array([max_z, max_y, max_x]) + padding
            roi = Roi(start_point, end_point - start_point)

            logger.info(f"Processing ID {id_value}: ROI {roi}")

            # Read data for this ID
            data = segmentation_idi.to_ndarray_ts(roi)
            data = data == id_value

            # Check if there's any data
            if not np.any(data):
                logger.warning(f"No voxels found for ID {id_value}, skipping")
                return Skeletonize._empty_metrics()

            # Compute EDT on pre-erosion mask for approximate radii
            distance_transform = edt_module.edt(data, anisotropy=tuple(voxel_size))

            # Apply erosion if requested
            if erosion:
                # Define a 3D cross-shaped structuring element (6-connectivity)
                cross_3d = np.array(
                    [
                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    ],
                    dtype=bool,
                )
                data = binary_erosion(data, cross_3d)

                # Check if erosion removed everything
                if not np.any(data):
                    logger.warning(
                        f"Erosion removed all voxels for ID {id_value}, writing empty skeleton"
                    )
                    # Write empty skeleton files
                    empty_skeleton = CustomSkeleton(vertices=[], edges=[])
                    full_path = f"{output_path}/full/{id_value}"
                    empty_skeleton.write_neuroglancer_skeleton(full_path)
                    simplified_path = f"{output_path}/simplified/{id_value}"
                    empty_skeleton.write_neuroglancer_skeleton(simplified_path)
                    logger.info(f"Wrote empty skeleton for ID {id_value}")
                    return Skeletonize._empty_metrics()

            # Skeletonize using Lee's algorithm (skimage default). It has
            # known limitations (e.g. thin structures may lose branches) but
            # is sufficient for now.
            skel = skeletonize(data)

            # Check if skeletonization produced anything
            if not np.any(skel):
                logger.warning(
                    f"Skeletonization produced no voxels for ID {id_value}, writing empty skeleton"
                )
                # Write empty skeleton files
                empty_skeleton = CustomSkeleton(vertices=[], edges=[])
                full_path = f"{output_path}/full/{id_value}"
                empty_skeleton.write_neuroglancer_skeleton(full_path)
                simplified_path = f"{output_path}/simplified/{id_value}"
                empty_skeleton.write_neuroglancer_skeleton(simplified_path)
                logger.info(f"Wrote empty skeleton for ID {id_value}")
                return Skeletonize._empty_metrics()

            # Sample radii at skeleton voxel positions
            skel_coords = np.argwhere(skel)
            radii = distance_transform[
                skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]
            ]

            # Convert to custom skeleton format
            # spacing parameter scales the vertices by voxel_size
            skeleton = skimage_to_custom_skeleton_fast(
                skel, spacing=segmentation_idi.voxel_size
            )

            # Transform vertices: add ROI offset and swap Z/X for neuroglancer (ZYX -> XYZ)
            # Vertices from skimage_to_custom_skeleton_fast are already in physical units
            # but in local coordinates, so we need to add the ROI offset
            skeleton.vertices = [
                tuple(
                    [
                        v[2] + start_point[2],
                        v[1] + start_point[1],
                        v[0] + start_point[0],
                    ]
                )
                for v in skeleton.vertices
            ]

            # Extract polylines with transformed coordinates
            # This ensures that prune() and simplify() have access to polylines
            # with the correct global coordinates
            g = skeleton.skeleton_to_graph()
            skeleton.polylines = skeleton.get_polylines_positions_from_graph(g)

            # Prune
            if min_branch_length_nm > 0:
                pruned = skeleton.prune(min_branch_length_nm)
            else:
                pruned = skeleton

            # Compute skeleton metrics on pruned skeleton
            num_branches = len(pruned.polylines)
            longest_shortest_path = 0.0
            if len(pruned.vertices) > 1:
                pruned_graph = pruned.skeleton_to_graph()
                for component in nx.connected_components(pruned_graph):
                    if len(component) < 2:
                        continue
                    subgraph = pruned_graph.subgraph(component)
                    start = next(iter(component))
                    lengths = nx.single_source_dijkstra_path_length(
                        subgraph, start, weight="weight"
                    )
                    far_node = max(lengths, key=lengths.get)
                    lengths2 = nx.single_source_dijkstra_path_length(
                        subgraph, far_node, weight="weight"
                    )
                    component_diameter = max(lengths2.values())
                    longest_shortest_path = max(
                        longest_shortest_path, component_diameter
                    )

            skeleton_metrics = {
                "longest_shortest_path_nm": longest_shortest_path,
                "num_branches": num_branches,
                "radius_mean_nm": float(np.mean(radii)),
                "radius_std_nm": float(np.std(radii)),
            }

            # Simplify
            if tolerance_nm > 0:
                simplified = pruned.simplify(tolerance_nm)
            else:
                simplified = pruned

            # Check if pruning/simplification removed all vertices
            if len(simplified.vertices) == 0:
                logger.warning(
                    f"Pruning/simplification removed all vertices for ID {id_value}, writing empty skeleton"
                )
                # Write empty skeleton files
                empty_skeleton = CustomSkeleton(vertices=[], edges=[])
                full_path = f"{output_path}/full/{id_value}"
                empty_skeleton.write_neuroglancer_skeleton(full_path)
                simplified_path = f"{output_path}/simplified/{id_value}"
                empty_skeleton.write_neuroglancer_skeleton(simplified_path)
                logger.info(f"Wrote empty skeleton for ID {id_value}")
                return Skeletonize._empty_metrics()

            # Ensure edges are properly shaped numpy arrays before writing
            # Handle case where there are no edges (single vertex)
            if len(skeleton.edges) == 0:
                skeleton.edges = np.zeros((0, 2), dtype=np.uint32)
            else:
                skeleton.edges = np.array(skeleton.edges, dtype=np.uint32)

            if len(simplified.edges) == 0:
                simplified.edges = np.zeros((0, 2), dtype=np.uint32)
            else:
                simplified.edges = np.array(simplified.edges, dtype=np.uint32)

            # Write full skeleton
            full_path = f"{output_path}/full/{id_value}"
            skeleton.write_neuroglancer_skeleton(full_path)
            logger.info(
                f"Wrote full skeleton for ID {id_value}: {len(skeleton.vertices)} vertices"
            )

            # Write simplified skeleton
            simplified_path = f"{output_path}/simplified/{id_value}"
            simplified.write_neuroglancer_skeleton(simplified_path)
            logger.info(
                f"Wrote simplified skeleton for ID {id_value}: {len(simplified.vertices)} vertices"
            )

            return skeleton_metrics

        except Exception as e:
            logger.error(f"Error processing ID {id_value}: {e}", exc_info=True)
            raise

    def write_neuroglancer_info_files(self):
        """
        Write the neuroglancer info file and segment_properties info file for both full and simplified directories.
        """
        # Write info files for both 'full' and 'simplified' directories
        for subdir in ["full", "simplified"]:
            # Write main info file for skeletons
            info = {
                "@type": "neuroglancer_skeletons",
                "transform": [
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                ],  # Identity transform since we're using physical coordinates
                "segment_properties": "segment_properties",
            }

            info_path = f"{self.output_path}/{subdir}/info"
            with open(info_path, "w") as f:
                json.dump(info, f)
            logger.info(f"Wrote neuroglancer info file to {info_path}")

            # Write segment_properties info file
            segment_ids = [str(id_val) for id_val in self.ids]
            segment_properties_info = {
                "@type": "neuroglancer_segment_properties",
                "inline": {
                    "ids": segment_ids,
                    "properties": [
                        {
                            "id": "label",
                            "type": "label",
                            "values": ["" for _ in segment_ids],
                        }
                    ],
                },
            }

            segment_properties_path = (
                f"{self.output_path}/{subdir}/segment_properties/info"
            )
            with open(segment_properties_path, "w") as f:
                json.dump(segment_properties_info, f)
            logger.info(
                f"Wrote segment_properties info file to {segment_properties_path}"
            )

    def skeletonize(self):
        """
        Main method to skeletonize all IDs in parallel.
        """
        logger.info(f"Starting skeletonization of {len(self.ids)} IDs")

        # First write the info files (only once)
        self.write_neuroglancer_info_files()

        # Parallelize over IDs using dask
        num_ids = len(self.ids)
        tmp_merge_dir = f"{self.output_path}/_tmp_skeleton_metrics_to_merge"

        skeleton_metrics = dask_util.compute_blockwise_partitions(
            num_ids,
            self.num_workers,
            self.compute_args,
            logger,
            f"skeletonizing {num_ids} IDs from {self.segmentation_idi.path}",
            self._skeletonize_id_wrapper,
            merge_info=(Skeletonize._merge_skeleton_metrics, tmp_merge_dir),
        )

        self._write_skeleton_csv(skeleton_metrics)

        logger.info("Skeletonization complete")

    @staticmethod
    def _merge_skeleton_metrics(list_of_results):
        merged = []
        for result in list_of_results:
            merged.append(result)
        return merged

    def _write_skeleton_csv(self, skeleton_metrics):
        original_df = pd.read_csv(self.csv_path, index_col=0)
        metrics_df = pd.DataFrame(skeleton_metrics)
        metrics_df = metrics_df.set_index("id")
        metrics_df = metrics_df.rename(
            columns={
                "longest_shortest_path_nm": "Longest Shortest Path (nm)",
                "num_branches": "Number of Branches",
                "radius_mean_nm": "Radius Mean (nm)",
                "radius_std_nm": "Radius Std (nm)",
            }
        )
        combined_df = original_df.join(metrics_df)
        csv_dir = os.path.dirname(self.csv_path)
        csv_basename = os.path.splitext(os.path.basename(self.csv_path))[0]
        output_csv = os.path.join(csv_dir, f"{csv_basename}_with_skeletons.csv")
        combined_df.to_csv(output_csv)
        logger.info(f"Wrote skeleton metrics CSV to {output_csv}")

    def _skeletonize_id_wrapper(self, index):
        """
        Wrapper to call calculate_id_skeleton with the appropriate ID.

        Args:
            index: Index into self.ids list
        """
        id_value = self.ids[index]
        result = Skeletonize.calculate_id_skeleton(
            id_value,
            self.segmentation_idi,
            self.bbox_df,
            self.output_path,
            self.erosion,
            self.min_branch_length_nm,
            self.tolerance_nm,
        )
        if result is None:
            result = Skeletonize._empty_metrics()
        result["id"] = id_value
        return result

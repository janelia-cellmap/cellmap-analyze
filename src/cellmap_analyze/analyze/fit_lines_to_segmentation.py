from funlib.persistence import Array, open_ds
from funlib.geometry import Roi
import numpy as np
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.dask_util import DaskBlock, create_blocks
from cellmap_analyze.util.io_util import (
    Timing_Messager,
    open_ds_tensorstore,
    print_with_datetime,
    split_dataset_path,
    to_ndarray_tensorstore,
)
from skimage import measure
from skimage.segmentation import expand_labels, find_boundaries
from sklearn.metrics.pairwise import pairwise_distances
from cellmap_analyze.util.bresenham3D import bresenham3DWithMask
import logging
from skimage.graph import pixel_graph
import networkx as nx
import dask.bag as db
import itertools
from funlib.segment.arrays import replace_values
import pandas as pd
from cellmap_analyze.util.zarr_util import create_multiscale_dataset
import dask.dataframe as dd
from funlib.geometry import Coordinate
import dask

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FitLinesToSegmentation:
    def __init__(self, input_csv, segmentation_ds_path, num_workers=8):
        self.df = pd.read_csv(input_csv)  # , nrows=1000)
        self.segmentation_ds = open_ds(*split_dataset_path(segmentation_ds_path))
        self.segmentation_ds_tensorstore = open_ds_tensorstore(segmentation_ds_path)
        self.voxel_size = self.segmentation_ds.voxel_size
        self.num_workers = num_workers

    @staticmethod
    def find_min_max_projected_points(points, line_point, line_direction):
        # chatgpt
        line_direction = line_direction / np.linalg.norm(
            line_direction
        )  # Normalize direction vector

        # Calculate the vector from line_point to each point
        point_vectors = points - line_point

        # Calculate the projection scalar for each point using dot product and broadcasting
        projection_scalars = np.sum(point_vectors * line_direction, axis=1)

        # Calculate the projected points for each point
        projected_points = (
            line_point + projection_scalars[:, np.newaxis] * line_direction
        )

        # Find the minimum and maximum projection scalar indices
        min_projection_idx = np.argmin(projection_scalars)
        max_projection_idx = np.argmax(projection_scalars)

        return (
            projected_points[min_projection_idx],
            projected_points[max_projection_idx],
        )

    @staticmethod
    def fit_line_to_points(points, voxel_size, offset, line_origin=0):
        # fit line to object voxels
        _, _, vv = np.linalg.svd(points - np.mean(points, axis=0), full_matrices=False)
        line_direction = vv[0]

        # find endpoints of line segment so that we can write it as neuroglancer annotations
        start_point, end_point = FitLinesToSegmentation.find_min_max_projected_points(
            points * voxel_size - voxel_size / 2 + offset,
            line_origin,
            line_direction,
        )

        return start_point, end_point

    @staticmethod
    def fit_line_to_object(data, id, voxel_size, offset, com):
        points = np.column_stack(np.where(data == id))
        start_point, end_point = FitLinesToSegmentation.fit_line_to_points(
            points, voxel_size, offset, com
        )
        return start_point, end_point

    def fit_lines_to_objects(self, df):
        results_df = []
        for _, row in df.iterrows():
            id = row["Object ID"]
            box_min = np.array([row[f"MIN {d} (nm)"] for d in ["X", "Y", "Z"]])
            box_max = np.array([row[f"MAX {d} (nm)"] for d in ["X", "Y", "Z"]])
            com = np.array([row[f"COM {d} (nm)"] for d in ["X", "Y", "Z"]])
            # define an roi to actually ecompass the bounding box
            roi = Roi(
                box_min - self.voxel_size, (box_max - box_min) + self.voxel_size * 2
            )
            data = to_ndarray_tensorstore(
                self.segmentation_ds_tensorstore,
                roi,
                self.voxel_size,
                Coordinate(self.segmentation_ds.roi.begin[::-1]),
            )
            line_start, line_end = FitLinesToSegmentation.fit_line_to_object(
                data, id, self.voxel_size, roi.offset, com=com
            )
            result_df = pd.DataFrame([row])

            for point_string, point_coords in zip(
                ["Start", "End"], [line_start, line_end]
            ):
                for dim_idx, dim in enumerate(["X", "Y", "Z"]):
                    result_df[f"Line {point_string} {dim} (nm)"] = point_coords[dim_idx]
            results_df.append(result_df)

        results_df = pd.concat(results_df, ignore_index=True)
        return results_df

    def get_fit_lines_to_objects(self):
        # append column with default values to df
        for s_e in ["Start", "End"]:
            for dim in ["X", "Y", "Z"]:
                self.df[f"Line {s_e} {dim} (nm)"] = np.NaN

        ddf = dd.from_pandas(self.df, npartitions=self.num_workers * 10)

        meta = pd.DataFrame(columns=self.df.columns)
        ddf_out = ddf.map_partitions(self.fit_lines_to_objects, meta=meta)
        with dask_util.start_dask(
            min(len(self.df), self.num_workers), "line fits", logger
        ):
            with io_util.Timing_Messager("Fitting lines", logger):
                results = ddf_out.compute()
        self.results = results

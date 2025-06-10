# %%
from funlib.geometry import Roi
import numpy as np
from cellmap_analyze.util.dask_util import (
    guesstimate_npartitions,
    start_dask,
)
from cellmap_analyze.util import io_util
from cellmap_analyze.util.image_data_interface import ImageDataInterface
import logging
import pandas as pd
import dask.dataframe as dd
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.neuroglancer_util import write_out_annotations
import fastremap
import cc3d

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FitLinesToSegmentations(ComputeConfigMixin):
    def __init__(
        self,
        input_csv,
        input_path,
        output_csv=None,
        output_annotations_dir=None,
        num_workers=8,
    ):
        super().__init__(num_workers)
        self.df = pd.read_csv(input_csv)  # , nrows=1000)
        self.segmentation_idi = ImageDataInterface(input_path)
        self.voxel_size = self.segmentation_idi.voxel_size
        self.num_workers = num_workers
        self.output_csv = output_csv
        if self.output_csv is None:
            self.output_csv = input_csv.replace(".csv", "_lines.csv")

        self.output_annotations_dir = output_annotations_dir

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
    def fit_line_to_points(points, voxel_size, offset, line_origin):
        # fit line to object voxels
        _, _, vv = np.linalg.svd(points - np.mean(points, axis=0), full_matrices=False)
        line_direction = vv[0]

        # find endpoints of line segment so that we can write it as neuroglancer annotations
        start_point, end_point = FitLinesToSegmentations.find_min_max_projected_points(
            points * voxel_size + offset,
            line_origin,
            line_direction,
        )

        return start_point, end_point

    @staticmethod
    def fit_line_to_object(data, id, voxel_size, offset):
        # only take largest component
        data = cc3d.connected_components(data == id, connectivity=6, binary_image=True)
        ids, counts = fastremap.unique(data[data > 0], return_counts=True)
        id = ids[np.argmax(counts)]
        points = np.column_stack(np.where(data == id))
        com = np.mean(points, axis=0) * voxel_size + offset
        start_point, end_point = FitLinesToSegmentations.fit_line_to_points(
            points, voxel_size, offset, com
        )
        return start_point, end_point

    def fit_lines_to_objects(self, df):
        results_df = []
        for _, row in df.iterrows():
            id = row["Object ID"]
            box_min = np.array([row[f"MIN {d} (nm)"] for d in ["Z", "Y", "X"]])
            box_max = np.array([row[f"MAX {d} (nm)"] for d in ["Z", "Y", "X"]])
            # define an roi to actually ecompass the bounding box
            roi = Roi(
                box_min - self.voxel_size, (box_max - box_min) + self.voxel_size * 2
            )
            data = self.segmentation_idi.to_ndarray_ts(roi)
            line_start, line_end = FitLinesToSegmentations.fit_line_to_object(
                data, id, self.voxel_size, roi.offset
            )
            result_df = pd.DataFrame([row])

            for point_string, point_coords in zip(
                ["Start", "End"], [line_start, line_end]
            ):
                for dim_idx, dim in enumerate(["Z", "Y", "X"]):
                    result_df[f"Line {point_string} {dim} (nm)"] = point_coords[dim_idx]
            results_df.append(result_df)

        results_df = pd.concat(results_df, ignore_index=True)
        return results_df

    def get_fit_lines_to_segmentations(self):
        # append column with default values to df
        for s_e in ["Start", "End"]:
            for dim in ["Z", "Y", "X"]:
                self.df[f"Line {s_e} {dim} (nm)"] = np.nan

        ddf = dd.from_pandas(
            self.df, npartitions=guesstimate_npartitions(self.df, self.num_workers)
        )

        meta = pd.DataFrame(columns=self.df.columns)
        ddf_out = ddf.map_partitions(self.fit_lines_to_objects, meta=meta)
        with start_dask(self.num_workers, "line fits", logger):
            with io_util.TimingMessager("Fitting lines", logger):
                # results = ddf_out.compute()
                df = ddf_out.compute(**self.compute_args)
        df["Object ID"] = df["Object ID"].astype(int)
        df.to_csv(self.output_csv, index=False)

        if self.output_annotations_dir is not None:
            with io_util.TimingMessager("Writing annotations", logger):
                cols = [f"Line Start {d} (nm)" for d in ["Z", "Y", "X"]] + [
                    f"Line End {d} (nm)" for d in ["Z", "Y", "X"]
                ]
                write_out_annotations(
                    self.output_annotations_dir,
                    df["Object ID"].values,
                    df[cols].to_numpy(),
                )

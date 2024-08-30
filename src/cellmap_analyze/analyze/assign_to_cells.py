from typing import Union, List
from cellmap_analyze.util.image_data_interface import (
    open_ds_tensorstore,
    to_ndarray_tensorstore,
)
import logging
import pandas as pd
import numpy as np
import os

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AssignToCells:
    def __init__(
        self,
        organelle_csvs: Union[str, List[str]],
        cell_ds_path: str,
        organelle_correction_offsets=None,
        base_resolution=None,
        cell_resolution=None,
    ):
        if isinstance(organelle_csvs, str):
            organelle_csvs = [organelle_csvs]

        self.organelle_info_dict = {}

        for organelle_csv in organelle_csvs:
            print(organelle_csv)
            df = pd.read_csv(organelle_csv)
            if "Total Objects" in df.columns:
                # delete last two columns of dataframe
                df = df.iloc[:, :-2]
            if organelle_correction_offsets is not None:
                AssignToCells.correct_coordinates(
                    df, organelle_correction_offsets[organelle_csv]
                )
            self.organelle_info_dict[organelle_csv] = df

        ds = open_ds_tensorstore(cell_ds_path)
        self.cell_domain = ds.domain
        print(self.cell_domain)
        self.base_resolution = base_resolution
        self.cell_resolution = cell_resolution
        if self.cell_resolution is None:
            self.cell_resolution = ds.spec().to_json()["metadata"]["resolution"][0]

        self.cell_data = to_ndarray_tensorstore(ds)

    @staticmethod
    def correct_coordinates(df, correction_offset):
        for measurement in ["COM", "MIN", "MAX"]:
            for coord in ["X", "Y", "Z"]:
                column = f"{measurement} {coord} (nm)"
                df[column] += correction_offset

    def assign_to_cells(self):
        for organelle_csv, df in self.organelle_info_dict.items():
            # get filename from organelle_csv
            filename = os.path.basename(organelle_csv)
            if filename == "cell.csv":
                continue
            df.insert(12, "Cell ID", 0)
            if "er.csv" in organelle_csv:
                df["Cell ID"] = df["Object ID"]
            else:
                inds = (
                    df[["COM Z (nm)", "COM Y (nm)", "COM X (nm)"]].to_numpy()
                    + self.base_resolution
                    / 2  # to shift top left pixel from (-base_resolution/2 to 0,0 so that the corners are aligned)
                ) // self.cell_resolution
                inds = inds.astype(int)

                in_bounds = np.all(
                    (inds >= self.cell_domain.inclusive_min), axis=1
                ) & np.all((inds < self.cell_domain.exclusive_max), axis=1)

                # out_of_bounds = np.logical_not(in_bounds)
                # df.loc[out_of_bounds, "Cell ID"] = 0

                df.loc[in_bounds, "Cell ID"] = self.cell_data[
                    inds[in_bounds, 0], inds[in_bounds, 1], inds[in_bounds, 2]
                ]
            df["Cell ID"] = df["Cell ID"].astype(int)

    # def correct_cell_csv(self, cell_csv, corrected_offset):
    #     df = pd.read_csv(cell_csv)
    #     if "Total Objects" in df.columns:
    #         # delete last two columns of dataframe
    #         df = df.iloc[:, :-2]
    #     AssignToCells.correct_coordinates(df, corrected_offset)
    #     self.organelle_info_dict[cell_csv] = df

    def write_updated_csvs(self, base_output_path):
        os.makedirs(base_output_path, exist_ok=True)
        for csv, df in self.organelle_info_dict.items():
            csv_name = os.path.basename(csv)
            output_path = base_output_path
            if "_to_" in csv_name:
                output_path = base_output_path + "/contact_sites/"
                os.makedirs(output_path, exist_ok=True)
            df.to_csv(output_path + "/" + csv_name, index=False)

import types
import numpy as np
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
    dask_computer,
    guesstimate_npartitions,
)
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.io_util import (
    get_name_from_path,
)
import pandas as pd
import logging
import dask.bag as db

from cellmap_analyze.util.measure_util import get_object_information

import os
from tqdm import tqdm
import time

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Measure:
    def __init__(
        self,
        input_path_1,
        input_path_2,
        roi=None,
        num_workers=10,
    ):
        self.input_idi_1 = ImageDataInterface(input_path_1)
        self.input_idi_2 = ImageDataInterface(input_path_2)

        self.contact_sites = False
        self.get_measurements_blockwise_extra_kwargs = {}
        output_voxel_size = min(
            self.input_idi_1.voxel_size, self.input_idi_2.voxel_size
        )
        self.input_idi_1.output_voxel_size = output_voxel_size
        self.input_idi_2.output_voxel_size = output_voxel_size

        self.num_workers = num_workers
        if roi is None:
            self.roi = self.input_idi_1.roi
        else:
            self.roi = roi
        self.num_workers = num_workers
        self.compute_args = {}
        if self.num_workers == 1:
            self.compute_args = {"scheduler": "single-threaded"}

    @staticmethod
    def get_overlapping_voxels_blockwise(
        block_index,
        input_idi_1: ImageDataInterface,
        input_idi_2: ImageDataInterface,
        roi,
    ):
        block = create_block_from_index(
            input_idi_1,
            block_index,
            roi=roi,
        )
        data_1 = input_idi_1.to_ndarray_ts(block.read_roi)
        data_2 = input_idi_2.to_ndarray_ts(block.read_roi)

        # get indices and values of overlapping voxels
        indices = np.where((data_1 > 0) & (data_2 > 0))
        z, y, x = indices
        values_1 = data_1[z, y, x]
        values_2 = data_2[z, y, x]
        block_offset = np.array(block.write_roi.begin)
        center_on_voxel = 0.5
        coms = (
            np.array([z, y, x]) + center_on_voxel + block_offset
        ) * input_idi_1.output_voxel_size
        # convert coms to tuple
        coms = list(map(tuple, coms.T))
        info_dict = dict(zip(coms, zip(values_1, values_2)))
        return object_informations

    @staticmethod
    def __summer(object_information_dicts):
        if type(object_information_dicts) is tuple:
            object_information_dicts = [object_information_dicts]
        elif isinstance(object_information_dicts, types.GeneratorType):
            object_information_dicts = list(object_information_dicts)

        for idx, object_information_dict in enumerate(tqdm(object_information_dicts)):
            if idx == 0:
                output_dict = object_information_dict
                continue

            for id, oi in object_information_dict.items():
                if id in output_dict:
                    output_dict[id] += oi
                else:
                    output_dict[id] = oi

        return output_dict

    def measure(self):
        num_blocks = dask_util.get_num_blocks(self.input_idi, self.roi)
        block_indexes = list(range(num_blocks))

        b = (
            db.from_sequence(
                block_indexes,
                npartitions=guesstimate_npartitions(block_indexes, self.num_workers),
            ).map(
                Measure.get_measurements_blockwise,
                self.input_idi,
                self.roi,
                self.global_offset,
                self.contact_sites,
                **self.get_measurements_blockwise_extra_kwargs,
            )
            # .reduction(Measure.__summer, Measure.__summer)
        )

        with dask_util.start_dask(
            self.num_workers,
            "measure object information",
            logger,
        ):
            with io_util.Timing_Messager("Measuring object information", logger):
                bagged_results = dask_computer(b, self.num_workers, **self.compute_args)

        # moved this out of dask, seems fast enough without having to daskify
        with io_util.Timing_Messager("Combining bagged results", logger):
            self.measurements = Measure.__summer(bagged_results)

    def write_measurements(self):
        os.makedirs(self.output_path, exist_ok=True)
        file_name = get_name_from_path(self.input_path)
        output_file = self.output_path + "/" + file_name + ".csv"

        # create dataframe
        columns = ["Object ID", "Volume (nm^3)", "Surface Area (nm^2)"]
        for category in ["COM", "MIN", "MAX"]:
            for d in ["X", "Y", "Z"]:
                columns.append(f"{category} {d} (nm)")

        if self.contact_sites:
            organelle_1_name = get_name_from_path(self.organelle_1_path)
            organelle_2_name = get_name_from_path(self.organelle_2_path)

            columns += [
                f"Contacting {organelle_1_name} IDs",
                f"Contacting {organelle_1_name} Surface Area (nm^2)",
                f"Contacting {organelle_2_name} IDs",
                f"Contacting {organelle_2_name} Surface Area (nm^2)",
            ]

        df = pd.DataFrame(
            index=np.arange(len(self.measurements)),
            columns=columns,
        )
        for i, (id, oi) in enumerate(self.measurements.items()):
            row = [
                id,
                oi.volume,
                oi.surface_area,
                *oi.com[::-1],
                *oi.bounding_box[:3][::-1],
                *oi.bounding_box[3:][::-1],
            ]
            if self.contact_sites:
                id_to_surface_area_dict_1 = (
                    oi.contacting_organelle_information_1.id_to_surface_area_dict
                )
                id_to_surface_area_dict_2 = (
                    oi.contacting_organelle_information_2.id_to_surface_area_dict
                )
                row += [
                    list(id_to_surface_area_dict_1.keys()),
                    list(id_to_surface_area_dict_1.values()),
                    list(id_to_surface_area_dict_2.keys()),
                    list(id_to_surface_area_dict_2.values()),
                ]
            df.loc[i] = row

        # ensure Object ID is written as an int
        df["Object ID"] = df["Object ID"].astype(int)
        df = df.sort_values(by=["Object ID"])
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_file, index=False)

    def get_measurements(self):
        self.measure()
        if self.output_path:
            with io_util.Timing_Messager("Writing object information", logger):
                self.write_measurements()

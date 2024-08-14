from funlib.persistence import open_ds
import numpy as np
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.dask_util import DaskBlock, create_blocks
from cellmap_analyze.util.information_holders import ObjectInformation
from cellmap_analyze.util.io_util import (
    get_name_from_path,
    open_ds_tensorstore,
    split_dataset_path,
    to_ndarray_tensorstore,
)
import pandas as pd
import logging
import dask.bag as db

from cellmap_analyze.util.measure_util import get_object_information

import os

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Measure:
    def __init__(
        self,
        input_path,
        output_path,
        roi=None,
        num_workers=10,
        **kwargs,
    ):
        self.input_path = input_path
        self.input_ds = open_ds(*split_dataset_path(self.input_path))
        self.input_ds_tensorstore = open_ds_tensorstore(self.input_path)
        self.output_path = output_path

        self.contact_sites = False
        if "organelle_1_path" in kwargs.keys() or "organelle_2_path" in kwargs.keys():
            if not (
                "organelle_1_path" in kwargs.keys()
                and "organelle_2_path" in kwargs.keys()
            ):
                raise ValueError(
                    "Must provide both organelle_1_path and organelle_2_path if doing contact site analysis"
                )
            self.organelle_1_path = kwargs["organelle_1_path"]
            self.organelle_2_path = kwargs["organelle_2_path"]
            self.organelle_1_ds = open_ds(*split_dataset_path(self.organelle_1_path))
            self.organelle_2_ds = open_ds(*split_dataset_path(self.organelle_2_path))
            self.organelle_1_ds_tensorstore = open_ds_tensorstore(self.organelle_1_path)
            self.organelle_2_ds_tensorstore = open_ds_tensorstore(self.organelle_2_path)
            self.contact_sites = True

        self.global_offset = np.zeros((3,))
        self.num_workers = num_workers
        if roi is None:
            self.roi = self.input_ds.roi
        else:
            self.roi = roi
        self.voxel_size = self.input_ds.voxel_size
        self.num_workers = num_workers
        self.client = client

    def get_measurements_blockwise(self, block: DaskBlock):
        data = to_ndarray_tensorstore(
            self.input_ds_tensorstore,
            block.read_roi,
            self.voxel_size,
            self.input_ds.roi.offset,
        )

        extra_kwargs = {}
        if self.contact_sites:
            extra_kwargs["organelle_1"] = to_ndarray_tensorstore(
                self.organelle_1_ds_tensorstore,
                block.read_roi,
                self.voxel_size,
                self.organelle_1_ds.roi.offset,
            )
            extra_kwargs["organelle_2"] = to_ndarray_tensorstore(
                self.organelle_2_ds_tensorstore,
                block.read_roi,
                self.voxel_size,
                self.organelle_2_ds.roi.offset,
            )

        # get information only from actual block(not including padding)
        block_offset = np.array(block.write_roi.begin) + self.global_offset
        object_informations = get_object_information(
            data, self.voxel_size[0], trim=1, offset=block_offset, **extra_kwargs
        )
        return object_informations

    def measure(self):
        def __summer(object_information_dicts):
            output_dict = {}
            for object_information_dict in object_information_dicts:
                for id, oi in object_information_dict.items():
                    if id in output_dict:
                        output_dict[id] += oi
                    else:
                        output_dict[id] = oi

            return output_dict

        b = (
            db.from_sequence(
                self.blocks, npartitions=min(len(self.blocks), self.num_workers * 10)
            )
            .map(self.get_measurements_blockwise)
            .reduction(__summer, __summer)
        )

        with dask_util.start_dask(
            self.num_workers,
            "measure object information",
            logger,
        ):
            with io_util.Timing_Messager("Measuring object information", logger):
                self.measurements = b.compute()

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
        df.to_csv(output_file, index=False)

    def get_measurements(self):
        self.blocks = create_blocks(
            self.roi, self.input_ds, padding=self.input_ds.voxel_size
        )
        self.measure()
        if self.output_path:
            self.write_measurements()

from funlib.persistence import open_ds
import numpy as np
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.dask_util import (
    DaskBlock,
    create_blocks,
    guesstimate_npartitions,
)
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
        self.get_measurements_blockwise_extra_kwargs = {}
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

            self.get_measurements_blockwise_extra_kwargs["organelle_1_ds"] = (
                self.organelle_1_ds
            )
            self.get_measurements_blockwise_extra_kwargs["organelle_2_ds"] = (
                self.organelle_2_ds
            )
            self.get_measurements_blockwise_extra_kwargs[
                "organelle_1_ds_tensorstore"
            ] = self.organelle_1_ds_tensorstore
            self.get_measurements_blockwise_extra_kwargs[
                "organelle_2_ds_tensorstore"
            ] = self.organelle_2_ds_tensorstore

            self.contact_sites = True

        self.global_offset = np.zeros((3,))
        self.num_workers = num_workers
        if roi is None:
            self.roi = self.input_ds.roi
        else:
            self.roi = roi
        self.voxel_size = self.input_ds.voxel_size
        self.num_workers = num_workers
        self.compute_args = {}
        if self.num_workers == 1:
            self.compute_args = {"scheduler": "single-threaded"}

    @staticmethod
    def get_measurements_blockwise(
        block: DaskBlock,
        input_ds,
        input_ds_tensorstore,
        voxel_size,
        global_offset,
        contact_sites,
        **kwargs,
    ):
        data = to_ndarray_tensorstore(
            input_ds_tensorstore,
            block.read_roi,
            voxel_size,
            input_ds.roi.offset,
        )

        extra_kwargs = {}
        if contact_sites:
            organelle_1_ds = kwargs.get("organelle_1_ds")
            organelle_1_ds_tensorstore = kwargs.get("organelle_1_ds_tensorstore")
            organelle_2_ds = kwargs.get("organelle_2_ds")
            organelle_2_ds_tensorstore = kwargs.get("organelle_2_ds_tensorstore")

            extra_kwargs["organelle_1"] = to_ndarray_tensorstore(
                organelle_1_ds_tensorstore,
                block.read_roi,
                voxel_size,
                organelle_1_ds.roi.offset,
            )
            extra_kwargs["organelle_2"] = to_ndarray_tensorstore(
                organelle_2_ds_tensorstore,
                block.read_roi,
                voxel_size,
                organelle_2_ds.roi.offset,
            )

        # get information only from actual block(not including padding)
        block_offset = np.array(block.write_roi.begin) + global_offset
        object_informations = get_object_information(
            data, voxel_size[0], trim=1, offset=block_offset, **extra_kwargs
        )
        return object_informations

    @staticmethod
    def __summer(object_information_dicts):
        output_dict = {}
        for object_information_dict in object_information_dicts:
            for id, oi in object_information_dict.items():
                if id in output_dict:
                    output_dict[id] += oi
                else:
                    output_dict[id] = oi

        return output_dict

    def measure(self):
        b = (
            db.from_sequence(
                self.blocks,
                npartitions=guesstimate_npartitions(self.blocks, self.num_workers),
            )
            .map(
                Measure.get_measurements_blockwise,
                self.input_ds,
                self.input_ds_tensorstore,
                self.voxel_size,
                self.global_offset,
                self.contact_sites,
                **self.get_measurements_blockwise_extra_kwargs,
            )
            .reduction(Measure.__summer, Measure.__summer)
        )

        with dask_util.start_dask(
            self.num_workers,
            "measure object information",
            logger,
        ):
            with io_util.Timing_Messager("Measuring object information", logger):
                self.measurements = b.compute(**self.compute_args)

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

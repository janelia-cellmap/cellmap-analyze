import numpy as np
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
)
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.information_holders import ObjectInformation
from cellmap_analyze.util.io_util import (
    get_name_from_path,
)
import pandas as pd
import logging

from cellmap_analyze.util.measure_util import get_object_information

import os

from cellmap_analyze.util.mixins import ComputeConfigMixin

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Measure(ComputeConfigMixin):
    def __init__(
        self,
        input_path,
        output_path,
        roi=None,
        num_workers=10,
        chunk_shape=None,
        **kwargs,
    ):
        super().__init__(num_workers)
        self.input_path = input_path
        self.input_idi = ImageDataInterface(self.input_path, chunk_shape=chunk_shape)
        self.output_path = output_path

        self.contact_sites = False
        self.get_measurements_blockwise_extra_kwargs = {}
        if (
            "organelle_1_path" in kwargs.keys() or "organelle_2_path" in kwargs.keys()
        ) and not (not kwargs["organelle_1_path"] and not kwargs["organelle_2_path"]):
            if not (
                "organelle_1_path" in kwargs.keys()
                and "organelle_2_path" in kwargs.keys()
            ):
                raise ValueError(
                    "Must provide both organelle_1_path and organelle_2_path if doing contact site analysis"
                )
            self.organelle_1_path = kwargs["organelle_1_path"]
            self.organelle_2_path = kwargs["organelle_2_path"]
            self.organelle_1_idi = ImageDataInterface(
                self.organelle_1_path, chunk_shape=chunk_shape
            )
            self.organelle_2_idi = ImageDataInterface(
                self.organelle_2_path, chunk_shape=chunk_shape
            )
            output_voxel_size = min(
                self.organelle_1_idi.voxel_size, self.organelle_2_idi.voxel_size
            )
            self.organelle_1_idi.output_voxel_size = output_voxel_size
            self.organelle_2_idi.output_voxel_size = output_voxel_size

            self.get_measurements_blockwise_extra_kwargs["organelle_1_idi"] = (
                self.organelle_1_idi
            )
            self.get_measurements_blockwise_extra_kwargs["organelle_2_idi"] = (
                self.organelle_2_idi
            )

            self.contact_sites = True
        self.output_directory = str(self.output_path) + "/measurements_to_merge/"
        self.global_offset = np.zeros((3,))
        self.num_workers = num_workers
        if roi is None:
            self.roi = self.input_idi.roi
        else:
            self.roi = roi
        self.voxel_size = self.input_idi.voxel_size

    @staticmethod
    def get_measurements_blockwise(
        block_index,
        input_idi: ImageDataInterface,
        roi,
        global_offset,
        contact_sites,
        **kwargs,
    ):
        block = create_block_from_index(
            input_idi,
            block_index,
            roi=roi,
            padding=input_idi.voxel_size[0],
        )
        data = input_idi.to_ndarray_ts(block.read_roi)

        extra_kwargs = {}
        if contact_sites:
            organelle_1_idi = kwargs.get("organelle_1_idi")
            organelle_2_idi = kwargs.get("organelle_2_idi")
            extra_kwargs["organelle_1"] = organelle_1_idi.to_ndarray_ts(
                block.read_roi,
            )
            extra_kwargs["organelle_2"] = organelle_2_idi.to_ndarray_ts(
                block.read_roi,
            )

        # get information only from actual block(not including padding)
        block_offset = np.array(block.write_roi.begin) + global_offset
        object_informations = get_object_information(
            data,
            input_idi.voxel_size[0],
            trim=1,
            offset=block_offset,
            **extra_kwargs,
        )

        # if not object_informations and not contact_sites:
        #     # NOTE: noticed that this code could hang and stop at 99% if processing a single organelle (non contact-site).
        #     # however if i added os.system calls to touch and rm file named with block index, then it wouldnt hang. so assuming that it has to do with a timing thing in that it only had to load/process a single dataset chunk.
        #     # sleeping for a little bit may therefore help:
        #     # NOTE: this happened after adding concurrency limits=1 when opening tensorstore, but never tested on complete single organelle datasets before that switch
        return object_informations

    @staticmethod
    def _merge_dicts(dicts) -> dict[int, ObjectInformation]:
        res = {}
        for current_dict in dicts:
            for k, v in current_dict.items():
                if k not in res:
                    res[k] = v
                else:
                    res[k] += v
        return res

    def measure(self):
        num_blocks = dask_util.get_num_blocks(self.input_idi, self.roi)
        self.measurements = dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            f"measuring blockwise object information for {self.input_idi.path}",
            Measure.get_measurements_blockwise,
            self.input_idi,
            self.roi,
            self.global_offset,
            self.contact_sites,
            **self.get_measurements_blockwise_extra_kwargs,
            merge_info=(Measure._merge_dicts, self.output_directory),
        )

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
        with io_util.Timing_Messager("Writing object information", logger):
            self.write_measurements()

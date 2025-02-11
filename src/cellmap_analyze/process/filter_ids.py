from typing import List, Union
import distributed
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
    split_dataset_path,
)
import logging
import dask.bag as db
import os
from cellmap_analyze.util.zarr_util import create_multiscale_dataset
import pandas as pd
from cellmap_analyze.util.block_util import relabel_block

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FilterIDs:
    def __init__(
        self,
        input_path,
        output_path=None,
        ids_to_keep: Union[List, str] = None,
        ids_to_remove: Union[List, str] = None,
        roi=None,
        num_workers=10,
    ):
        # must have either ids_to_keep or ids_to_remove not both
        if ids_to_keep is None and ids_to_remove is None:
            raise ValueError("Must provide either ids_to_keep or ids_to_remove")
        if ids_to_keep is not None and ids_to_remove is not None:
            raise ValueError(
                "Must provide either ids_to_keep or ids_to_remove not both"
            )

        self.ids_to_keep = ids_to_keep
        self.ids_to_remove = ids_to_remove
        if self.ids_to_keep:
            if type(self.ids_to_keep) == str:
                if os.path.exists(self.ids_to_keep):
                    self.ids_to_keep = pd.read_csv(self.ids_to_keep)[
                        "Object IDs"
                    ].tolist()
                else:
                    self.ids_to_keep = [int(i) for i in self.ids_to_keep.split(",")]
            new_dtype = np.min_scalar_type(len(self.ids_to_keep))
        if self.ids_to_remove:
            raise NotImplementedError("ids_to_remove not implemented yet")

        self.relabeling_dict = dict(
            zip(self.ids_to_keep, range(1, len(self.ids_to_keep) + 1))
        )

        self.input_path = input_path
        self.input_idi = ImageDataInterface(self.input_path)
        if roi is None:
            self.roi = self.input_idi.roi
        else:
            self.roi = roi
        self.voxel_size = self.input_idi.voxel_size

        self.output_path = output_path
        if self.output_path is None:
            self.output_path = self.input_path

        output_ds_name = get_name_from_path(self.output_path)
        output_ds_basepath = split_dataset_path(self.output_path)[0]
        self.output_ds_path = f"{output_ds_basepath}/{output_ds_name}_filteredIDs"

        create_multiscale_dataset(
            self.output_ds_path,
            dtype=new_dtype,
            voxel_size=self.voxel_size,
            total_roi=self.roi,
            write_size=self.input_idi.chunk_shape * self.voxel_size,
        )
        self.output_idi = ImageDataInterface(
            self.output_ds_path + "/s0",
            mode="r+",
        )

        self.num_workers = num_workers
        self.compute_args = {}
        if self.num_workers == 1:
            self.compute_args = {"scheduler": "single-threaded"}
            self.num_local_threads_available = 1
            self.local_config = None
        else:
            self.num_local_threads_available = len(os.sched_getaffinity(0))
            self.local_config = {
                "jobqueue": {
                    "local": {
                        "ncpus": self.num_local_threads_available,
                        "processes": self.num_local_threads_available,
                        "cores": self.num_local_threads_available,
                        "log-directory": "job-logs",
                        "name": "dask-worker",
                    }
                }
            }

    @staticmethod
    def filter_ids_blockwise(block_index, input_idi, output_idi, relabeling_dict):
        block = create_block_from_index(
            input_idi,
            block_index,
        )
        if isinstance(relabeling_dict, distributed.Future):
            global_relabeling_dict = relabeling_dict.result()
        else:
            global_relabeling_dict = relabeling_dict
        relabel_block(
            block,
            input_idi,
            output_idi,
            global_relabeling_dict=global_relabeling_dict,
        )

    def filter_ids(self):
        num_blocks = dask_util.get_num_blocks(self.input_idi)
        block_indexes = list(range(num_blocks))

        with dask_util.start_dask(
            self.num_workers,
            "filter ids",
            logger,
        ) as client:
            if client is not None:
                # Client is none if doing testing
                relabeling_dict = client.scatter(self.relabeling_dict, broadcast=True)
            else:
                relabeling_dict = self.relabeling_dict

            b = db.from_sequence(
                block_indexes,
                npartitions=guesstimate_npartitions(block_indexes, self.num_workers),
            ).map(
                FilterIDs.filter_ids_blockwise,
                self.input_idi,
                self.output_idi,
                relabeling_dict,
            )
            with io_util.Timing_Messager("Filtering IDs", logger):
                dask_computer(b, self.num_workers, **self.compute_args)

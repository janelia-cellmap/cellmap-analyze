from typing import List, Union
import numpy as np
from cellmap_analyze.process.connected_components import ConnectedComponents
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.io_util import get_output_path_from_input_path
import logging
import os
import pandas as pd
import shutil

from cellmap_analyze.util.mixins import ComputeConfigMixin

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FilterIDs(ComputeConfigMixin):
    def __init__(
        self,
        input_path,
        output_path=None,
        ids_to_keep: Union[List, str] = None,
        ids_to_remove: Union[List, str] = None,
        binarize=False,
        roi=None,
        num_workers=10,
        chunk_shape=None,
    ):
        super().__init__(num_workers)
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
                        "Object ID"
                    ].tolist()
                else:
                    self.ids_to_keep = [int(i) for i in self.ids_to_keep.split(",")]
            if binarize:
                self.new_dtype = np.uint8
            else:
                self.new_dtype = np.min_scalar_type(len(self.ids_to_keep))
        if self.ids_to_remove:
            raise NotImplementedError("ids_to_remove not implemented yet")

        if binarize:
            self.global_relabeling_dict = dict(
                zip(self.ids_to_keep, [1] * len(self.ids_to_keep))
            )
        else:
            self.global_relabeling_dict = dict(
                zip(self.ids_to_keep, range(1, len(self.ids_to_keep) + 1))
            )

        self.input_path = input_path
        self.input_idi = ImageDataInterface(self.input_path, chunk_shape=chunk_shape)
        if roi is None:
            self.roi = self.input_idi.roi
        else:
            self.roi = roi
        self.voxel_size = self.input_idi.voxel_size

        if output_path is None:
            output_path = get_output_path_from_input_path(input_path, "_filteredIDs")
        self.output_path = str(output_path).rstrip("/")

        # Use helper function to generate relabeling dict path (handles root datasets correctly)
        self.relabeling_dict_path = get_output_path_from_input_path(
            self.output_path, "_relabeling_dict"
        )

    def get_filtered_ids(self):
        ConnectedComponents.write_memmap_relabeling_dicts(
            self.global_relabeling_dict, self.relabeling_dict_path
        )
        ConnectedComponents.relabel_dataset(
            self.input_idi,
            self.output_path,
            self.roi,
            self.new_dtype,
            self.relabeling_dict_path,
            self.num_workers,
            self.compute_args,
        )
        shutil.rmtree(self.relabeling_dict_path)

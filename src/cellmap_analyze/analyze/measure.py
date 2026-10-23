# %%
import numpy as np
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
)
from cellmap_analyze.util.grid_util import crop_to_roi, grid_of, voxel_corner
from cellmap_analyze.util.information_holders import ObjectInformation
from cellmap_analyze.util.io_util import (
    get_leaf_name_from_path,
)
import pandas as pd
import logging

from cellmap_analyze.util.measure_util import get_object_information
from cellmap_analyze.util.zarr_util import open_dataset
import os

from cellmap_analyze.util.mixins import ComputeConfigMixin

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _principal_axis_values(oi):
    """The eight principal-axis columns for one object, in the CSV's order."""
    lengths, vectors = oi.principal_axes
    return [*lengths, oi.elongation, oi.flatness, *vectors[:, 0][::-1]]


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
        self.input_vol = open_dataset(self.input_path)
        # Measured on the input's voxels (within roi); every other dataset is
        # read onto them.
        self.grid = crop_to_roi(grid_of(self.input_vol, chunk_shape), roi)
        self.output_path = str(output_path).rstrip("/")

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
            self.organelle_1_vol = open_dataset(self.organelle_1_path)
            self.organelle_2_vol = open_dataset(self.organelle_2_path)

            self.get_measurements_blockwise_extra_kwargs["organelle_1_vol"] = (
                self.organelle_1_vol
            )
            self.get_measurements_blockwise_extra_kwargs["organelle_2_vol"] = (
                self.organelle_2_vol
            )

            self.contact_sites = True

        if "raw_path" in kwargs and kwargs["raw_path"]:
            self.raw_path = kwargs["raw_path"]
            # If raw_path is a multiscale group (no scale level given), pick
            # the level whose voxel size best matches the input segmentation
            # so intensities line up with minimal resampling.
            # Intensities, so when no level matches, a finer one is block-averaged
            # to the segmentation's voxel size rather than sampled one voxel per block.
            self.raw_vol = open_dataset(
                self.raw_path, voxel_size=self.input_vol.voxel_size, method="linear"
            )
            self.get_measurements_blockwise_extra_kwargs["raw_vol"] = self.raw_vol

        # Handle root datasets (empty name) by using "data" as default
        input_name = get_leaf_name_from_path(self.input_path)
        if not input_name:
            input_name = "data"
        self.output_directory = (
            str(self.output_path) + f"/measurements_to_merge_{input_name}/"
        )
        self.global_offset = np.zeros((3,))
        self.num_workers = num_workers
        self.voxel_size = self.input_vol.voxel_size

    @staticmethod
    def pad_with_face_neighbor_blocks(
        vol, grid, block, return_none_if_main_block_empty=False
    ):
        """The block's voxels of ``grid``, read from ``vol``, with the one-voxel-thick
        faces of its six neighbours around them (the corners and edges stay 0); reads
        past the data are filled."""
        main_block = vol.read_onto(grid, block.begin, block.end)
        if return_none_if_main_block_empty and not np.any(main_block):
            return None
        data = np.zeros(np.array(main_block.shape) + 2, dtype=main_block.dtype)
        data[1:-1, 1:-1, 1:-1] = main_block
        interior = [slice(1, -1)] * 3

        for d in range(3):
            neg_begin, neg_end = list(block.begin), list(block.end)
            neg_begin[d], neg_end[d] = block.begin[d] - 1, block.begin[d]
            pos_begin, pos_end = list(block.begin), list(block.end)
            pos_begin[d], pos_end[d] = block.end[d], block.end[d] + 1

            # build slice objects dynamically
            neg_idx = interior.copy()
            neg_idx[d] = slice(0, 1)  # face at the “minus” side
            data[tuple(neg_idx)] = vol.read_onto(grid, neg_begin, neg_end)

            pos_idx = interior.copy()
            pos_idx[d] = slice(-1, None)  # face at the “plus” side
            data[tuple(pos_idx)] = vol.read_onto(grid, pos_begin, pos_end)

        return data

    @staticmethod
    def get_measurements_blockwise(
        block_index,
        input_vol,
        grid,
        global_offset,
        contact_sites,
        **kwargs,
    ):
        block = create_block_from_index(grid, block_index)
        # main block
        data = Measure.pad_with_face_neighbor_blocks(
            input_vol,
            grid,
            block,
            return_none_if_main_block_empty=True,
        )
        if data is None:
            return {}

        extra_kwargs = {}
        if contact_sites:
            # The contact sites were found on the organelles' finer voxels, so
            # reading the organelles onto the contact sites' grid is lossless.
            extra_kwargs["organelle_1"] = Measure.pad_with_face_neighbor_blocks(
                kwargs.get("organelle_1_vol"), grid, block
            )
            extra_kwargs["organelle_2"] = Measure.pad_with_face_neighbor_blocks(
                kwargs.get("organelle_2_vol"), grid, block
            )

        raw_vol = kwargs.get("raw_vol")
        if raw_vol is not None:
            # Read raw at the block (no face-neighbor padding needed for intensity).
            # raw_valid marks voxels that came from padding/no-overlap fill
            # rather than a real read, so they can be excluded from intensity
            # stats instead of being silently counted as real intensity 0.
            raw_data, raw_valid = raw_vol.read_onto(
                grid, block.begin, block.end, mask=True
            )
            extra_kwargs["raw_data"] = raw_data
            extra_kwargs["raw_valid"] = raw_valid

        # get information only from actual block(not including padding)
        block_offset = np.array(voxel_corner(grid, block.begin)) + global_offset
        object_informations = get_object_information(
            data,
            input_vol.voxel_size,
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
        num_blocks = dask_util.get_num_blocks(self.grid)
        self.measurements = dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            f"measuring blockwise object information for {self.input_vol.level_path} and putting it here: {self.output_directory}",
            Measure.get_measurements_blockwise,
            self.input_vol,
            self.grid,
            self.global_offset,
            self.contact_sites,
            **self.get_measurements_blockwise_extra_kwargs,
            merge_info=(Measure._merge_dicts, self.output_directory),
        )

    def write_measurements(self):
        os.makedirs(self.output_path, exist_ok=True)
        file_name = get_leaf_name_from_path(self.input_path)
        # Handle root datasets (empty name) by using "measurements" as filename
        if not file_name:
            file_name = "measurements"
        output_file = self.output_path + "/" + file_name + ".csv"

        # create dataframe
        columns = [
            "Object ID",
            "Volume (nm^3)",
            "Surface Area (nm^2)",
            "Radius of Gyration (nm)",
        ]
        for category in ["COM", "MIN", "MAX"]:
            for d in ["X", "Y", "Z"]:
                columns.append(f"{category} {d} (nm)")

        # Principal axes, from the second-moment tensor accumulated blockwise:
        # the lengths size an object and their ratios say how elongated or flat
        # it is, while the major axis vector says which way it points.
        columns += [
            "Principal Axis Length 1 (nm)",
            "Principal Axis Length 2 (nm)",
            "Principal Axis Length 3 (nm)",
            "Elongation",
            "Flatness",
            "Principal Axis 1 X",
            "Principal Axis 1 Y",
            "Principal Axis 1 Z",
        ]

        self.has_raw_intensity = any(
            oi.has_raw_intensity for oi in self.measurements.values()
        )
        if self.has_raw_intensity:
            columns += ["Mean Intensity", "Std Intensity"]

        if self.contact_sites:
            organelle_1_name = get_leaf_name_from_path(self.organelle_1_path)
            organelle_2_name = get_leaf_name_from_path(self.organelle_2_path)
            # Handle root datasets (empty names) by using defaults
            if not organelle_1_name:
                organelle_1_name = "organelle_1"
            if not organelle_2_name:
                organelle_2_name = "organelle_2"

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
                oi.radius_of_gyration,
                *oi.com[::-1],
                *oi.bounding_box[:3][::-1],
                *oi.bounding_box[3:][::-1],
                *_principal_axis_values(oi),
            ]
            if self.has_raw_intensity:
                row += [oi.mean_intensity, oi.std_intensity]
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
        with io_util.TimingMessager("Writing object information", logger):
            self.write_measurements()

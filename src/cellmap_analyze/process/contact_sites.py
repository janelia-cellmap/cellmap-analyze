from funlib.persistence import open_ds
from funlib.geometry import Roi
import numpy as np
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.dask_util import create_blocks
from cellmap_analyze.util.io_util import split_dataset_path
from skimage import measure
from skimage.segmentation import find_boundaries

# from cellmap_analyze.util.bresenham3D import bresenham3DWithMask
from cellmap_analyze.cythonizing.bresenham3D import bresenham3DWithMask
import logging
from cellmap_analyze.process.connected_components import ConnectedComponents
from scipy.spatial import KDTree
import dask.bag as db
from cellmap_analyze.util.tmp_analysis_util import trim_array
from cellmap_analyze.util.io_util import open_ds_tensorstore, to_ndarray_tensorstore
from funlib.persistence import open_ds
from cellmap_analyze.util.zarr_util import (
    create_multiscale_dataset,
)


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ContactSites:
    def __init__(
        self,
        organelle_1_path,
        organelle_2_path,
        output_path,
        contact_distance_nm=30,
        minimum_volume_nm_3=None,
        num_workers=10,
        roi=None,
    ):

        self.organelle_1_tensorstore = open_ds_tensorstore(organelle_1_path)
        self.organelle_2_tensorstore = open_ds_tensorstore(organelle_2_path)
        self.organelle_1 = open_ds(*split_dataset_path(organelle_1_path))
        self.organelle_2 = open_ds(*split_dataset_path(organelle_2_path))

        self.rescale_factor = (
            self.organelle_1.voxel_size[0] / self.organelle_2.voxel_size[0]
        )

        if self.organelle_1.voxel_size != self.organelle_2.voxel_size:
            raise ValueError("Voxel sizes of organelles do not match")
        self.voxel_size = self.organelle_1.voxel_size

        if self.organelle_1.roi != self.organelle_2.roi:
            raise ValueError("ROIs of organelles do not match")

        self.contact_distance_voxels = (
            contact_distance_nm / self.organelle_1.voxel_size[0]
        )

        self.padding_voxels = int(np.ceil(self.contact_distance_voxels) + 1)
        # add one to ensure accuracy during surface area calculation since we need to make sure that neighboring ones are calculated

        self.roi = roi
        if self.roi is None:
            self.roi = self.organelle_1.roi

        self.output_path = output_path
        self.output_path_blockwise = output_path + "_blockwise"

        if minimum_volume_nm_3 is None:
            minimum_volume_nm_3 = (
                np.pi * (contact_distance_nm**2)
            ) * contact_distance_nm

        self.minimum_volume_nm_3 = minimum_volume_nm_3
        self.num_workers = num_workers
        self.voxel_volume = np.prod(self.voxel_size)
        self.voxel_face_area = self.voxel_size[1] * self.voxel_size[2]

        self.contact_sites_blockwise_ds = create_multiscale_dataset(
            self.output_path_blockwise,
            dtype=np.uint64,
            voxel_size=self.voxel_size,
            total_roi=self.roi,
            write_size=self.organelle_1.chunk_shape * self.organelle_1.voxel_size,
        )

    @staticmethod
    def get_surface_voxels(organelle_1, organelle_2):
        surface_voxels_1 = find_boundaries(organelle_1, mode="inner")
        surface_voxels_2 = find_boundaries(organelle_2, mode="inner")

        return (
            surface_voxels_1,
            surface_voxels_2,
        )

    @staticmethod
    def get_ndarray_contact_sites(
        organelle_1, organelle_2, contact_distance_voxels, mask_out_surface_voxels=False
    ):
        # bresenham method
        surface_voxels_1, surface_voxels_2 = ContactSites.get_surface_voxels(
            organelle_1, organelle_2
        )
        if mask_out_surface_voxels:
            mask = (organelle_1 > 0) | (organelle_2 > 0)
        else:
            # use nonsurface voxels as mask
            mask = ((organelle_1 > 0) & (surface_voxels_1 == 0)) | (
                (organelle_2 > 0) & (surface_voxels_2 == 0)
            )
        overlap_voxels = np.argwhere((organelle_1 > 0) & (organelle_2 > 0))
        del organelle_1, organelle_2

        current_pair_contact_voxels = np.zeros_like(surface_voxels_1, np.uint64)

        # get all voxel pairs that are within the contact distance
        # get npwhere result as array of coordinates

        object_1_surface_voxel_coordinates = np.argwhere(surface_voxels_1)
        object_2_surface_voxel_coordinates = np.argwhere(surface_voxels_2)
        del surface_voxels_1, surface_voxels_2

        # Create KD-trees for efficient distance computation
        tree1 = KDTree(object_1_surface_voxel_coordinates)
        tree2 = KDTree(object_2_surface_voxel_coordinates)

        # Find all pairs of points from both organelles within the threshold distance
        contact_voxels_list_of_lists = tree1.query_ball_tree(
            tree2, contact_distance_voxels
        )

        all_valid_voxels = set()
        for i, sublist in enumerate(contact_voxels_list_of_lists):
            for j in sublist:
                contact_voxel_1 = object_1_surface_voxel_coordinates[i]
                contact_voxel_2 = object_2_surface_voxel_coordinates[j]

                if mask_out_surface_voxels:
                    mask[contact_voxel_1[0], contact_voxel_1[1], contact_voxel_1[2]] = 0
                    mask[contact_voxel_2[0], contact_voxel_2[1], contact_voxel_2[2]] = 0
                valid_voxels = bresenham3DWithMask(
                    *contact_voxel_1, *contact_voxel_2, mask=mask
                )

                if valid_voxels:
                    all_valid_voxels.update(valid_voxels)

                if mask_out_surface_voxels:
                    mask[contact_voxel_1[0], contact_voxel_1[1], contact_voxel_1[2]] = 1
                    mask[contact_voxel_2[0], contact_voxel_2[1], contact_voxel_2[2]] = 1

        if len(all_valid_voxels) > 0:
            x_coords, y_coords, z_coords = zip(*all_valid_voxels)
            current_pair_contact_voxels[x_coords, y_coords, z_coords] = 1

        if len(overlap_voxels) > 0:
            indices = tuple(zip(*overlap_voxels))
            current_pair_contact_voxels[indices] = 1

        # need connectivity of 3 due to bresenham allowing diagonals
        current_pair_contact_sites = measure.label(
            current_pair_contact_voxels, connectivity=3
        )

        return current_pair_contact_sites.astype(np.uint64)

    def calculate_block_contact_sites(self, block: dask_util.DaskBlock):
        organelle_1 = to_ndarray_tensorstore(
            self.organelle_1_tensorstore,
            block.read_roi,
            self.voxel_size,
            self.organelle_1.roi.offset,
        )
        organelle_2 = to_ndarray_tensorstore(
            self.organelle_2_tensorstore,
            block.read_roi,
            self.voxel_size,
            self.organelle_1.roi.offset,
        )
        global_id_offset = ConnectedComponents.convert_position_to_global_id(
            block.write_roi.get_begin(), self.organelle_1.roi.shape
        )
        contact_sites = ContactSites.get_ndarray_contact_sites(
            organelle_1, organelle_2, self.contact_distance_voxels
        )

        contact_sites[contact_sites > 0] += global_id_offset
        self.contact_sites_blockwise_ds[block.write_roi] = trim_array(
            contact_sites, self.padding_voxels
        )

    def calculate_contact_sites_blockwise(self):

        b = db.from_sequence(
            self.blocks, npartitions=min(len(self.blocks), self.num_workers * 10)
        ).map(self.calculate_block_contact_sites)

        with dask_util.start_dask(
            self.num_workers,
            "calculate contact sites",
            logger,
        ):
            with io_util.Timing_Messager("Calculating contact sites", logger):
                b.compute()

    def get_contact_sites(self):
        self.blocks = create_blocks(
            self.roi,
            self.organelle_1,
            padding=self.organelle_1.voxel_size * self.padding_voxels,
        )
        self.calculate_contact_sites_blockwise()

        cc = ConnectedComponents(
            tmp_blockwise_ds_path=self.output_path_blockwise + "/s0",
            output_ds_path=self.output_path,
            roi=self.roi,
            num_workers=self.num_workers,
            minimum_volume_nm_3=self.minimum_volume_nm_3,
            connectivity=3,
        )
        cc.merge_connected_components_across_blocks()

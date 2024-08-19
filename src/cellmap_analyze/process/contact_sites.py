from funlib.persistence import open_ds
from funlib.geometry import Roi
import numpy as np
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
    guesstimate_npartitions,
)
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.io_util import get_name_from_path
from skimage import measure
from cellmap_analyze.cythonizing.process_arrays import initialize_contact_site_array

# from cellmap_analyze.util.bresenham3D import bresenham3DWithMask
from cellmap_analyze.cythonizing.bresenham3D import (
    bresenham3DWithMask,
    bresenham3DWithMaskSingle,
)
import logging
from cellmap_analyze.process.connected_components import ConnectedComponents
from scipy.spatial import KDTree
import dask.bag as db
from cellmap_analyze.util.measure_util import trim_array
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
        self.organelle_1_idi = ImageDataInterface(organelle_1_path)
        self.organelle_2_idi = ImageDataInterface(organelle_2_path)
        output_voxel_size = min(
            self.organelle_1_idi.voxel_size, self.organelle_2_idi.voxel_size
        )
        self.organelle_2_idi.output_voxel_size = output_voxel_size
        self.organelle_1_idi.output_voxel_size = output_voxel_size
        self.voxel_size = output_voxel_size
        self.contact_distance_voxels = (
            contact_distance_nm / self.organelle_1_idi.voxel_size[0]
        )

        self.padding_voxels = int(np.ceil(self.contact_distance_voxels) + 1)
        # add one to ensure accuracy during surface area calculation since we need to make sure that neighboring ones are calculated

        self.roi = roi
        if self.roi is None:
            self.roi = self.organelle_1_idi.roi

        if not get_name_from_path(output_path):
            output_path = (
                output_path
                + f"/{get_name_from_path(organelle_1_path)}_to_{get_name_from_path(organelle_2_path)}"
            )

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

        create_multiscale_dataset(
            self.output_path_blockwise,
            dtype=np.uint64,
            voxel_size=self.voxel_size,
            total_roi=self.roi,
            write_size=self.organelle_1_idi.chunk_shape
            * self.organelle_1_idi.voxel_size,
        )

        self.contact_sites_blockwise_idi = ImageDataInterface(
            self.output_path_blockwise + "/s0", mode="r+"
        )

        self.compute_args = {}
        if self.num_workers == 1:
            self.compute_args = {"scheduler": "single-threaded"}

    # @staticmethod
    # def get_surface_voxels(organelle_1, organelle_2):
    #     surface_voxels_1 = find_boundary(organelle_1, mode="inner") > 0
    #     surface_voxels_2 = find_boundary(organelle_2, mode="inner") > 0

    #     return (
    #         surface_voxels_1,
    #         surface_voxels_2,
    #     )

    @staticmethod
    def get_ndarray_contact_sites(
        organelle_1, organelle_2, contact_distance_voxels, mask_out_surface_voxels=False
    ):
        surface_voxels_1 = np.zeros_like(organelle_1, np.uint8)
        surface_voxels_2 = np.zeros_like(organelle_2, np.uint8)
        mask = np.zeros_like(organelle_1, np.uint8)
        current_pair_contact_sites = np.zeros_like(organelle_1, np.uint64)
        initialize_contact_site_array(
            organelle_1,
            organelle_2,
            surface_voxels_1,
            surface_voxels_2,
            mask,
            current_pair_contact_sites,
            mask_out_surface_voxels,
        )

        # bresenham method

        # find_boundary(organelle_1, surface_voxels_1)
        # find_boundary(organelle_2, surface_voxels_2)

        # organelle_1 = organelle_1 > 0
        # organelle_2 = organelle_2 > 0
        # if mask_out_surface_voxels:
        #     mask = organelle_1 | organelle_2
        # else:
        #     # use nonsurface voxels as mask
        #     mask = (organelle_1 & ~surface_voxels_1) | (organelle_2 & ~surface_voxels_2)
        # mask = mask.astype(np.uint8)

        # overlap_voxels = np.argwhere(organelle_1 & organelle_2)
        # if len(overlap_voxels) > 0:
        #     indices = tuple(zip(*overlap_voxels))
        #     current_pair_contact_sites[indices] = 1
        #     found_contact_voxels = True

        del organelle_1, organelle_2

        # # get all voxel pairs that are within the contact distance
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
                if (
                    current_pair_contact_sites[
                        contact_voxel_1[0], contact_voxel_1[1], contact_voxel_1[2]
                    ]
                    or current_pair_contact_sites[
                        contact_voxel_2[0], contact_voxel_2[1], contact_voxel_2[2]
                    ]
                ):
                    continue

                # if np.all(np.abs(contact_voxel_1 - contact_voxel_2) <= 1):
                #     continue

                valid_voxels = bresenham3DWithMaskSingle(
                    *contact_voxel_1,
                    *contact_voxel_2,
                    mask=mask,
                )
                if valid_voxels:
                    all_valid_voxels.update(valid_voxels)

        if mask_out_surface_voxels:
            mask[contact_voxel_1[0], contact_voxel_1[1], contact_voxel_1[2]] = 1
            mask[contact_voxel_2[0], contact_voxel_2[1], contact_voxel_2[2]] = 1

        found_contact_voxels = False
        if len(all_valid_voxels) > 0:
            x_coords, y_coords, z_coords = zip(*all_valid_voxels)
            current_pair_contact_sites[x_coords, y_coords, z_coords] = 1
            found_contact_voxels = True

        if found_contact_voxels:
            # need connectivity of 3 due to bresenham allowing diagonals
            current_pair_contact_sites = measure.label(
                current_pair_contact_sites, connectivity=3
            )

        return current_pair_contact_sites.astype(np.uint64)

    @staticmethod
    def calculate_block_contact_sites(
        block_index,
        organelle_1_idi: ImageDataInterface,
        organelle_2_idi: ImageDataInterface,
        contact_sites_blockwise_idi: ImageDataInterface,
        contact_distance_voxels,
        padding_voxels,
    ):
        block = create_block_from_index(
            contact_sites_blockwise_idi,
            block_index,
            padding=padding_voxels * contact_sites_blockwise_idi.voxel_size[0],
        )
        organelle_1 = organelle_1_idi.to_ndarray_ts(block.read_roi)
        organelle_2 = organelle_2_idi.to_ndarray_ts(block.read_roi)
        global_id_offset = ConnectedComponents.convert_position_to_global_id(
            block.write_roi.begin / contact_sites_blockwise_idi.voxel_size,
            contact_sites_blockwise_idi.roi.shape
            / contact_sites_blockwise_idi.voxel_size,
        )
        contact_sites = ContactSites.get_ndarray_contact_sites(
            organelle_1, organelle_2, contact_distance_voxels
        )

        contact_sites[contact_sites > 0] += global_id_offset
        contact_sites_blockwise_idi.ds[block.write_roi] = trim_array(
            contact_sites, padding_voxels
        )

    def calculate_contact_sites_blockwise(self):
        num_blocks = dask_util.get_num_blocks(self.contact_sites_blockwise_idi)
        block_indexes = list(range(num_blocks))
        b = db.from_sequence(
            block_indexes,
            npartitions=guesstimate_npartitions(block_indexes, self.num_workers),
        ).map(
            ContactSites.calculate_block_contact_sites,
            self.organelle_1_idi,
            self.organelle_2_idi,
            self.contact_sites_blockwise_idi,
            self.contact_distance_voxels,
            self.padding_voxels,
        )

        with dask_util.start_dask(
            self.num_workers,
            "calculate contact sites",
            logger,
        ):
            with io_util.Timing_Messager("Calculating contact sites", logger):
                b.compute(**self.compute_args)

    def get_contact_sites(self):
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

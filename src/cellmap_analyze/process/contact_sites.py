import numpy as np
from cellmap_analyze.util import dask_util
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
)
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.io_util import get_name_from_path, get_output_path_from_input_path
from cellmap_analyze.cythonizing.process_arrays import initialize_contact_site_array
from cellmap_analyze.cythonizing.bresenham3D import bresenham_3D_lines
import logging
from cellmap_analyze.process.connected_components import ConnectedComponents
from scipy.spatial import KDTree
from cellmap_analyze.util.measure_util import trim_array
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.zarr_util import (
    create_multiscale_dataset_idi,
)
import cc3d

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ContactSites(ComputeConfigMixin):
    def __init__(
        self,
        organelle_1_path,
        organelle_2_path,
        output_path,
        contact_distance_nm=30,
        minimum_volume_nm_3=None,
        num_workers=10,
        roi=None,
        chunk_shape=None,
    ):
        super().__init__(num_workers)
        self.organelle_1_idi = ImageDataInterface(
            organelle_1_path, chunk_shape=chunk_shape
        )
        self.organelle_2_idi = ImageDataInterface(
            organelle_2_path, chunk_shape=chunk_shape
        )
        output_voxel_size = min(
            self.organelle_1_idi.voxel_size, self.organelle_2_idi.voxel_size
        )
        self.organelle_2_idi.output_voxel_size = output_voxel_size
        self.organelle_1_idi.output_voxel_size = output_voxel_size
        self.voxel_size = output_voxel_size

        self.contact_distance_voxels = contact_distance_nm / output_voxel_size[0]

        self.padding_voxels = int(np.ceil(self.contact_distance_voxels) + 1)
        # add one to ensure accuracy during surface area calculation since we need to make sure that neighboring ones are calculated

        self.roi = roi
        if self.roi is None:
            self.roi = self.organelle_1_idi.roi.intersect(self.organelle_2_idi.roi)

        if not get_name_from_path(output_path):
            output_path = (
                output_path
                + f"/{get_name_from_path(organelle_1_path)}_{get_name_from_path(organelle_2_path)}_contacts"
            )

        self.output_path = str(output_path).rstrip("/")

        if minimum_volume_nm_3 is None:
            minimum_volume_nm_3 = (
                np.pi * ((contact_distance_nm / 2) ** 2)
            ) * contact_distance_nm

        self.minimum_volume_nm_3 = minimum_volume_nm_3
        self.num_workers = num_workers
        self.voxel_volume = np.prod(self.voxel_size)
        self.voxel_face_area = self.voxel_size[1] * self.voxel_size[2]

        # Use helper function to generate blockwise path (handles root datasets correctly)
        blockwise_path = get_output_path_from_input_path(output_path, "_blockwise")

        self.contact_sites_blockwise_idi = create_multiscale_dataset_idi(
            blockwise_path,
            dtype=np.uint64,
            voxel_size=self.voxel_size,
            total_roi=self.roi,
            write_size=self.organelle_1_idi.chunk_shape * self.voxel_size,
        )

    @staticmethod
    def get_ndarray_contact_sites(
        organelle_1,
        organelle_2,
        contact_distance_voxels,
        mask_out_surface_voxels=False,
        zero_pad=False,
    ):
        if zero_pad:
            organelle_1 = np.pad(organelle_1, 1)
            organelle_2 = np.pad(organelle_2, 1)

        surface_voxels_1 = np.zeros_like(organelle_1, np.uint8)
        surface_voxels_2 = np.zeros_like(organelle_2, np.uint8)
        mask = np.zeros_like(organelle_1, np.uint8)
        current_pair_contact_sites = np.zeros_like(organelle_1, np.uint8)
        initialize_contact_site_array(
            organelle_1,
            organelle_2,
            surface_voxels_1,
            surface_voxels_2,
            mask,
            current_pair_contact_sites,
            mask_out_surface_voxels,
        )

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

        found_contact_voxels = bresenham_3D_lines(
            contact_voxels_list_of_lists,
            object_1_surface_voxel_coordinates,
            object_2_surface_voxel_coordinates,
            current_pair_contact_sites,
            2 * np.ceil(contact_distance_voxels),
            mask,
        )

        if found_contact_voxels:
            # need connectivity of 3 due to bresenham allowing diagonals
            current_pair_contact_sites = cc3d.connected_components(
                current_pair_contact_sites,
                connectivity=26,
                binary_image=True,
            )

        if zero_pad:
            return current_pair_contact_sites[1:-1, 1:-1, 1:-1].astype(np.uint64)
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
        if not np.any(organelle_1):
            # if organelle_1 is empty, we can skip this block
            contact_sites_blockwise_idi.ds[block.write_roi] = trim_array(
                np.zeros(organelle_1.shape, dtype=np.uint64), padding_voxels
            )
            return
        organelle_2 = organelle_2_idi.to_ndarray_ts(block.read_roi)
        if not np.any(organelle_2):
            # if organelle_2 is empty, we can skip this block
            contact_sites_blockwise_idi.ds[block.write_roi] = trim_array(
                np.zeros(organelle_1.shape, dtype=np.uint64), padding_voxels
            )
            return
        global_id_offset = block_index * np.prod(
            block.full_block_size / contact_sites_blockwise_idi.voxel_size[0],
            dtype=np.uint64,
        )  # have to use full_block_size since before if we use write_roi, blocks on the end will be smaller and will have incorrect offsets
        contact_sites = ContactSites.get_ndarray_contact_sites(
            organelle_1, organelle_2, contact_distance_voxels
        )
        contact_sites[contact_sites > 0] += global_id_offset
        contact_sites_blockwise_idi.ds[block.write_roi] = trim_array(
            contact_sites, padding_voxels
        )

    def calculate_contact_sites_blockwise(self):
        num_blocks = dask_util.get_num_blocks(
            self.contact_sites_blockwise_idi, self.roi
        )
        dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            f"calculating blockwise contact sites between {self.organelle_1_idi.path} and {self.organelle_2_idi.path}",
            ContactSites.calculate_block_contact_sites,
            self.organelle_1_idi,
            self.organelle_2_idi,
            self.contact_sites_blockwise_idi,
            self.contact_distance_voxels,
            self.padding_voxels,
        )

    def get_contact_sites(self):
        self.calculate_contact_sites_blockwise()

        cc = ConnectedComponents(
            connected_components_blockwise_path=self.contact_sites_blockwise_idi.path,
            output_path=self.output_path,
            roi=self.roi,
            num_workers=self.num_workers,
            minimum_volume_nm_3=self.minimum_volume_nm_3,
            connectivity=3,
            delete_tmp=True,
        )
        cc.merge_connected_components_across_blocks()

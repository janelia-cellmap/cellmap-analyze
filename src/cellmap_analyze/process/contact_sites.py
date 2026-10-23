import numpy as np
from cellmap_analyze.util import dask_util
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
)
from cellmap_analyze.util.grid_util import crop_to_roi, intersection
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.io_util import get_leaf_name_from_path, get_output_path_from_input_path
from cellmap_analyze.cythonizing.process_arrays import initialize_contact_site_array
from cellmap_analyze.cythonizing.bresenham3D import bresenham_3D_lines
import logging
from cellmap_analyze.process.connected_components import ConnectedComponents
from scipy.spatial import KDTree
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
        # Both organelles are read onto one grid: the element-wise minimum voxel
        # size, over the box both cover (or the voxels of it in roi)
        self.voxel_size = tuple(
            min(v1, v2)
            for v1, v2 in zip(
                self.organelle_1_idi.voxel_size, self.organelle_2_idi.voxel_size
            )
        )
        self.roi = roi
        self.grid = crop_to_roi(
            intersection(
                [self.organelle_1_idi.grid, self.organelle_2_idi.grid],
                self.voxel_size,
                chunks=self.organelle_1_idi.chunk_shape,
            ),
            roi,
        )
        self.organelle_1_idi.output_grid = self.grid
        self.organelle_2_idi.output_grid = self.grid

        # Store contact distance in nm for physical distance calculations
        self.contact_distance_nm = contact_distance_nm

        # Use minimum voxel size across all dimensions
        min_voxel_size = float(min(self.voxel_size))
        self.contact_distance_voxels = float(contact_distance_nm / min_voxel_size)

        self.padding_voxels = int(np.ceil(self.contact_distance_voxels) + 1)
        # add one to ensure accuracy during surface area calculation since we need to make sure that neighboring ones are calculated

        if not get_leaf_name_from_path(output_path):
            output_path = (
                output_path
                + f"/{get_leaf_name_from_path(organelle_1_path)}_{get_leaf_name_from_path(organelle_2_path)}_contacts"
            )

        self.output_path = str(output_path).rstrip("/")

        if minimum_volume_nm_3 is None:
            minimum_volume_nm_3 = (
                np.pi * ((contact_distance_nm / 2) ** 2)
            ) * contact_distance_nm

        self.minimum_volume_nm_3 = minimum_volume_nm_3
        self.num_workers = num_workers
        self.voxel_volume = float(np.prod(self.voxel_size))
        self.voxel_face_area = float(self.voxel_size[1] * self.voxel_size[2])

        # Use helper function to generate blockwise path (handles root datasets correctly)
        blockwise_path = get_output_path_from_input_path(output_path, "_blockwise")

        self.contact_sites_blockwise_idi = create_multiscale_dataset_idi(
            blockwise_path, dtype=np.uint64, grid=self.grid
        )

    @staticmethod
    def get_ndarray_contact_sites(
        organelle_1,
        organelle_2,
        contact_distance_voxels,
        mask_out_surface_voxels=False,
        zero_pad=False,
        voxel_size=None,
        contact_distance_nm=None,
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

        # For anisotropic data, scale coordinates to physical space
        if voxel_size is not None and contact_distance_nm is not None:
            # Scale voxel coordinates by voxel size to get physical coordinates
            object_1_physical = object_1_surface_voxel_coordinates * voxel_size
            object_2_physical = object_2_surface_voxel_coordinates * voxel_size
            # Create KD-trees in physical space
            tree1 = KDTree(object_1_physical)
            tree2 = KDTree(object_2_physical)
            # Find pairs within physical distance threshold
            contact_voxels_list_of_lists = tree1.query_ball_tree(
                tree2, contact_distance_nm
            )
        else:
            # Legacy behavior for isotropic data or when voxel_size not provided
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
        voxel_size,
        contact_distance_nm,
    ):
        block = create_block_from_index(
            contact_sites_blockwise_idi,
            block_index,
            padding=padding_voxels,
        )
        organelle_1 = organelle_1_idi.to_ndarray_ts(block.read_begin, block.read_end)
        if not np.any(organelle_1):
            # if organelle_1 is empty, we can skip this block
            contact_sites_blockwise_idi.write(block.begin, block.end, 0)
            return
        organelle_2 = organelle_2_idi.to_ndarray_ts(block.read_begin, block.read_end)
        if not np.any(organelle_2):
            # if organelle_2 is empty, we can skip this block
            contact_sites_blockwise_idi.write(block.begin, block.end, 0)
            return
        # the full block shape, so the smaller blocks at the far ends get their own ids too
        global_id_offset = block.id * np.prod(
            contact_sites_blockwise_idi.chunk_shape, dtype=np.uint64
        )
        contact_sites = ContactSites.get_ndarray_contact_sites(
            organelle_1,
            organelle_2,
            contact_distance_voxels,
            voxel_size=voxel_size,
            contact_distance_nm=contact_distance_nm,
        )
        contact_sites[contact_sites > 0] += global_id_offset
        contact_sites_blockwise_idi.write(
            block.begin, block.end, contact_sites[block.inner]
        )

    def calculate_contact_sites_blockwise(self):
        num_blocks = dask_util.get_num_blocks(self.contact_sites_blockwise_idi)
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
            self.voxel_size,
            self.contact_distance_nm,
        )

    def get_contact_sites(self):
        self.calculate_contact_sites_blockwise()

        # the blockwise contact sites are already on the grid, roi and all
        cc = ConnectedComponents(
            connected_components_blockwise_path=self.contact_sites_blockwise_idi.path,
            output_path=self.output_path,
            num_workers=self.num_workers,
            minimum_volume_nm_3=self.minimum_volume_nm_3,
            connectivity=3,
            delete_tmp=True,
        )
        cc.merge_connected_components_across_blocks()

from funlib.persistence import open_ds
from funlib.geometry import Roi
import numpy as np
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.dask_util import create_blocks
from cellmap_analyze.util.io_util import (
    Timing_Messager,
    print_with_datetime,
    split_dataset_path,
)
from skimage import measure
from skimage.segmentation import expand_labels, find_boundaries
from sklearn.metrics.pairwise import pairwise_distances
from skimage.morphology import erosion

# from cellmap_analyze.util.bresenham3D import bresenham3DWithMask
from cellmap_analyze.cythonizing.bresenham3D import bresenham3DWithMask
import logging
from cellmap_analyze.process.connected_components import ConnectedComponents
import pandas as pd
from scipy.spatial import KDTree
import skfmm
from numpy import ma
from cellmap_analyze.util.bresenhamline import bresenhamline, bresenhamline_with_mask
import dask.bag as db
from cellmap_analyze.util.analysis_util import (
    trim_array,
    calculate_surface_areas_voxelwise,
    get_region_properties,
)


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ContactingOrganelleInformation:
    def __init__(self, id_to_surface_area_dict={}):
        self.id_to_surface_area_dict = id_to_surface_area_dict

    @staticmethod
    def combine_id_to_surface_area_dicts(dict1, dict2):
        # make dict1 the larger dict
        if len(dict1) < len(dict2):
            dict1, dict2 = dict2, dict1

        dict1 = dict1.copy()
        for id, surface_area in dict2.items():
            dict1[id] = dict1.get(id, 0) + surface_area
        return dict1

    def __iadd__(self, other: "ContactingOrganelleInformation"):
        self.id_to_surface_area_dict = (
            ContactingOrganelleInformation.combine_id_to_surface_area_dicts(
                self.id_to_surface_area_dict, other.id_to_surface_area_dict
            )
        )
        return self


class ContactSiteInformation:
    def __init__(
        self,
        volume: float = 0,
        surface_area: float = 0,
        com: np.ndarray = np.array([0, 0, 0]),
        id_to_surface_area_dict_1: dict = {},
        id_to_surface_area_dict_2: dict = {},
        bounding_box: list = [np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf],
    ):

        self.volume = volume
        self.surface_area = surface_area
        self.contacting_organelle_information_1 = ContactingOrganelleInformation(
            id_to_surface_area_dict_1
        )
        self.contacting_organelle_information_2 = ContactingOrganelleInformation(
            id_to_surface_area_dict_2
        )
        self.bounding_box = bounding_box
        self.com = com

    def __iadd__(self, other: "ContactSiteInformation"):
        self.com = ((self.com * self.volume) + (other.com * other.volume)) / (
            self.volume + other.volume
        )
        self.volume += other.volume
        self.surface_area += other.surface_area
        self.contacting_organelle_information_1 += (
            other.contacting_organelle_information_1
        )
        self.contacting_organelle_information_2 += (
            other.contacting_organelle_information_2
        )

        ndim = len(self.com)
        new_bounding_box = [
            min(self.bounding_box[d], other.bounding_box[d]) for d in range(ndim)
        ]
        new_bounding_box += [
            max(self.bounding_box[d + ndim], other.bounding_box[d + ndim])
            for d in range(ndim)
        ]
        self.bounding_box = new_bounding_box
        return self


class ContactSites:
    def __init__(
        self,
        organelle_1_path,
        organelle_2_path,
        contact_distance_nm=30,
        minimum_volume_nm_3=20_000,
        num_workers=10,
        roi=None,
    ):

        self.organelle_1 = open_ds(*split_dataset_path(organelle_1_path))
        self.organelle_2 = open_ds(*split_dataset_path(organelle_2_path))

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

        self.minimum_volume_voxels = minimum_volume_nm_3 / np.prod(self.voxel_size)
        self.num_workers = num_workers
        self.voxel_volume = np.prod(self.voxel_size)
        self.voxel_face_area = self.voxel_size[1] * self.voxel_size[2]

    @staticmethod
    def get_surface_voxels(organelle_1, organelle_2):
        surface_voxels_1 = find_boundaries(
            organelle_1, mode="inner"
        )  # organelle_1 * find_boundaries(organelle_1, mode="inner")
        surface_voxels_2 = find_boundaries(
            organelle_2, mode="inner"
        )  # organelle_2 * find_boundaries(organelle_2, mode="inner")

        return (
            surface_voxels_1,
            surface_voxels_2,
        )

    @staticmethod
    def get_ndarray_contact_sites_bresenham(
        object_1_surface_voxels,
        object_2_surface_voxels,
        contact_distance_voxels,
        overlap_voxels,
        mask=None,
    ):
        print_with_datetime("get_current_pair_contact_sites", logger)
        current_pair_contact_voxels = np.zeros_like(object_1_surface_voxels, np.uint64)

        # get all voxel pairs that are within the contact distance
        # get npwhere result as array of coordinates

        object_1_surface_voxel_coordinates = np.argwhere(object_1_surface_voxels)
        object_2_surface_voxel_coordinates = np.argwhere(object_2_surface_voxels)

        # Create KD-trees for efficient distance computation
        tree1 = KDTree(object_1_surface_voxel_coordinates)
        tree2 = KDTree(object_2_surface_voxel_coordinates)

        # Find all pairs of points from both organelles within the threshold distance
        contact_voxels_list_of_lists = tree1.query_ball_tree(
            tree2, contact_distance_voxels
        )
        contact_voxels_pairs = [
            [i, j]
            for i, sublist in enumerate(contact_voxels_list_of_lists)
            for j in sublist
        ]
        contact_voxels_pairs = np.array(contact_voxels_pairs).T
        print_with_datetime("got contact voxels", logger)

        contact_voxels_1 = object_1_surface_voxel_coordinates[contact_voxels_pairs[0]]
        contact_voxels_2 = object_2_surface_voxel_coordinates[contact_voxels_pairs[1]]
        if len(overlap_voxels) > 0:
            indices = tuple(zip(*overlap_voxels))
            current_pair_contact_voxels[indices] = 1

        print_with_datetime("do bresenham", logger)
        all_valid_voxels = set()
        for contact_voxel_1, contact_voxel_2 in zip(contact_voxels_1, contact_voxels_2):
            valid_voxels = bresenham3DWithMask(
                *contact_voxel_1, *contact_voxel_2, mask=mask
            )

            if valid_voxels:
                all_valid_voxels.update(valid_voxels)
        x_coords, y_coords, z_coords = zip(*all_valid_voxels)
        current_pair_contact_voxels[x_coords, y_coords, z_coords] = 1
        print_with_datetime("bresenham done", logger)

        # need connectivity of 3 due to bresenham allowing diagonals
        current_pair_contact_sites = measure.label(
            current_pair_contact_voxels, connectivity=3
        )

        return current_pair_contact_sites.astype(np.uint64)

    @staticmethod
    def get_ndarray_contact_site_information(
        organelle_1,
        organelle_2,
        contact_distance_voxels,
    ):
        surface_voxels_1, surface_voxels_2 = ContactSites.get_surface_voxels(
            organelle_1, organelle_2
        )
        # use nonsurface voxels as mask
        mask = ((organelle_1 > 0) & (surface_voxels_1 == 0)) | (
            (organelle_2 > 0) & (surface_voxels_2 == 0)
        )

        # outputs
        df = pd.DataFrame(
            columns=[
                "object_1_ids",
                "object_2_ids",
                "global_blockwise_contact_site",
            ]
        )
        overlap_voxels = np.argwhere((organelle_1 > 0) & (organelle_2 > 0))
        contact_sites = ContactSites.get_ndarray_contact_sites_bresenham(
            surface_voxels_1,
            surface_voxels_2,
            contact_distance_voxels,
            overlap_voxels,
            mask,
        )

        if np.any(contact_sites):
            current_contact_site_ids = np.unique(contact_sites[contact_sites > 0])

            # Note some contact site ids may be overwritten but that shouldnt be an issue
            for current_contact_site_id in current_contact_site_ids:
                current_contact_site = contact_sites == current_contact_site_id
                new_row = {
                    "object_1_ids": list(np.unique(organelle_1[current_contact_site])),
                    "object_2_ids": list(np.unique(organelle_2[current_contact_site])),
                    "global_blockwise_contact_site": current_contact_site_id,
                }
                df.loc[len(df)] = new_row

        return contact_sites, df

    @staticmethod
    def get_contacting_organelle_information(
        contact_sites, contacting_organelle, voxel_face_area=1, trim=0
    ):

        surface_areas = calculate_surface_areas_voxelwise(
            contacting_organelle, voxel_face_area
        )

        # trim so we are only considering current block
        surface_areas = trim_array(surface_areas, trim)
        contact_sites = trim_array(contact_sites, trim)
        contacting_organelle = trim_array(contacting_organelle, trim)

        # limit looking to only where contact sites overlap with objects
        mask = np.logical_and(contact_sites > 0, contacting_organelle > 0)
        contact_sites = contact_sites[mask].ravel()
        contacting_organelle = contacting_organelle[mask].ravel()

        surface_areas = surface_areas[mask].ravel()
        groups, counts = np.unique(
            np.array([contact_sites, contacting_organelle, surface_areas]),
            axis=1,
            return_counts=True,
        )
        contact_site_ids = groups[0]
        contacting_ids = groups[1]
        surface_areas = groups[2] * counts
        contact_site_to_contacting_information_dict = {}
        for contact_site_id, contacting_id, surface_area in zip(
            contact_site_ids, contacting_ids, surface_areas
        ):
            coi = contact_site_to_contacting_information_dict.get(
                contact_site_id,
                ContactingOrganelleInformation(),
            )
            coi += ContactingOrganelleInformation({contacting_id: surface_area})
            contact_site_to_contacting_information_dict[contact_site_id] = coi
        return contact_site_to_contacting_information_dict

    @staticmethod
    def get_contacting_organelles_information(
        contact_sites, organelle_1, organelle_2, trim=0
    ):
        contacting_organelle_information_1 = (
            ContactSites.get_contacting_organelle_information(
                contact_sites, organelle_1, trim=trim
            )
        )
        contacting_organelle_information_2 = (
            ContactSites.get_contacting_organelle_information(
                contact_sites, organelle_2, trim=trim
            )
        )
        return contacting_organelle_information_1, contacting_organelle_information_2

    def get_block_contact_site_information(self, block: dask_util.DaskBlock):
        print_with_datetime(f"Calculating contact site information for", logger)
        organelle_1 = self.organelle_1.to_ndarray(block.roi)
        organelle_2 = self.organelle_2.to_ndarray(block.roi)
        print_with_datetime(f"ndarray done {block.roi}, {block.id}", logger)

        contact_sites, df = ContactSites.get_ndarray_contact_site_information(
            organelle_1, organelle_2, self.contact_distance_voxels
        )

        print_with_datetime(f"blockwise_contact_sites done", logger)

        print_with_datetime(f"region props", logger)
        # get which objects from organelle_1 and organelle_2 overlap with which ids from blockwise_contact_sites
        # get surface area of organelle 1
        # voxel_surface_area
        # voxel_volume
        global_id_offset = ConnectedComponents.convertPositionToGlobalID(
            block.roi.get_begin() / self.voxel_size, organelle_1.shape
        )
        self.region_props = get_region_properties(
            contact_sites,
            self.voxel_face_area,
            self.voxel_volume,
            trim=self.padding_voxels,
        )
        if self.region_props is None:
            return None
        self.region_props["ID"] += global_id_offset
        contact_sites[contact_sites > 0] += global_id_offset
        print_with_datetime(f"region props done", logger)
        (
            self.contacting_organelle_information_1,
            self.contacting_organelle_information_2,
        ) = ContactSites.get_contacting_organelles_information(
            contact_sites,
            organelle_1,
            organelle_2,
            trim=self.padding_voxels,
        )
        # return organelle_1, organelle_2, contact_sites
        print_with_datetime(f"contacting organelle information done", logger)
        return organelle_1, organelle_2, contact_sites
        csi = {}
        for _, region_prop in self.region_props.iterrows():
            id = region_prop["ID"]
            # print_with_datetime(
            #     f"{id}, {self.contacting_organelle_information_1}", logger
            # )
            csi[id] = ContactSiteInformation(
                volume=region_prop["Volume (nm^3)"],
                surface_area=region_prop["Surface Area (nm^2)"],
                com=region_prop[["COM Z (nm)", "COM Y (nm)", "COM X (nm)"]].to_numpy(),
                # if the id is outside of the non-paded crop it wont exist in the following dicts
                id_to_surface_area_dict_1=self.contacting_organelle_information_1.get(
                    id, {}
                ),
                id_to_surface_area_dict_2=self.contacting_organelle_information_2.get(
                    id, {}
                ),
            )

        return csi
        # write out blockwise_contact_site

    def get_contact_site_information_with_dask(self):
        b = db.from_sequence(self.blocks, npartitions=self.num_workers * 10).map(
            self.get_block_contact_site_information
        )

        with dask_util.start_dask(
            self.num_workers,
            "calculate contact site information",
            logger,
        ):
            with io_util.Timing_Messager(
                "Calculating contact site information", logger
            ):
                self.blockwise_results = b.compute()

    def calculate_contact_sites_with_dask(self):
        self.blocks = create_blocks(
            self.roi,
            self.organelle_1,
            padding=self.organelle_1.voxel_size * self.padding_voxels,
        )
        self.num_workers = min(self.num_workers, len(self.blocks))
        csi = self.get_contact_site_information_with_dask()
        return csi

    def dfer():
        df["block_id", "object_1", "object_2", "global_blockwise_contact_site"]

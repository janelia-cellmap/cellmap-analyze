from funlib.persistence import open_ds
from funlib.geometry import Roi
import numpy as np
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
from cellmap_analyze.util.bresenham3d import bresenham3DWithMask
import logging
from cellmap_analyze.process.connected_components import ConnectedComponents
import pandas as pd
from scipy.spatial import KDTree
import skfmm
from numpy import ma
from cellmap_analyze.util.bresenhamline import bresenhamline, bresenhamline_with_mask


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
        contact_distance_nm=30,
        minimum_volume_nm_3=20_000,
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

        self.roi = roi
        if self.roi is None:
            self.roi = self.organelle_1.roi

        self.blocks = create_blocks(
            self.roi, self.organelle_1, padding=np.ceil(self.contact_distance_voxels)
        )

    @staticmethod
    def get_contact_boundaries(organelle_1, organelle_2, contact_distance_voxels):
        print_with_datetime("Contact boundary 1", logger)
        # contact_boundary_1 = (
        #     expand_labels(organelle_1, distance=contact_distance_voxels) - organelle_1
        # )
        # print_with_datetime("Contact boundary 2", logger)
        # contact_boundary_2 = (
        #     expand_labels(organelle_2, distance=contact_distance_voxels) - organelle_2
        # )
        contact_boundary_1 = [1]
        contact_boundary_2 = [2]

        print_with_datetime("Find boundaries", logger)
        surface_voxels_1 = organelle_1 * find_boundaries(organelle_1, mode="inner")
        surface_voxels_2 = organelle_2 * find_boundaries(organelle_2, mode="inner")

        return (
            contact_boundary_1,
            contact_boundary_2,
            surface_voxels_1,
            surface_voxels_2,
        )

    @staticmethod
    def tmp_get_current_pair_contact_sites(
        object_1_surface_voxels,
        object_2_surface_voxels,
        contact_distance_voxels,
        overlap_voxels,
        mask=None,
        method="geodesic",
    ):
        if method == "geodesic":
            return ContactSites.get_current_pair_contact_sites_geodesic(
                object_1_surface_voxels,
                object_2_surface_voxels,
                contact_distance_voxels,
                overlap_voxels,
                mask,
            )
        else:
            return ContactSites.get_current_pair_contact_sites_bresenham(
                object_1_surface_voxels,
                object_2_surface_voxels,
                contact_distance_voxels,
                overlap_voxels,
                mask,
            )

    @staticmethod
    def get_current_pair_contact_sites_geodesic(
        object_1,
        object_2,
        contact_distance_voxels,
        overlap_voxels,
        mask=None,
    ):
        current_pair_contact_voxels = np.zeros_like(object_1_surface_voxels, np.uint64)

        organelle_masked = ma.masked_array(
            organelle_2 == 0,
            mask=(organelle_1 > 0) | (interior_2 > 0),
        )

        if len(overlap_voxels) > 0:
            indices = tuple(zip(*overlap_voxels))
            current_pair_contact_voxels[indices] = 1
        for contact_voxel_1, contact_voxel_2 in zip(contact_voxels_1, contact_voxels_2):
            valid_voxels = bresenham3DWithMask(
                *contact_voxel_1, *contact_voxel_2, mask=mask
            )
            if valid_voxels:
                indices = tuple(zip(*valid_voxels))
                current_pair_contact_voxels[indices] = 1

        current_pair_contact_sites = measure.label(
            current_pair_contact_voxels, connectivity=3
        )
        return current_pair_contact_sites.astype(np.uint64)

    @staticmethod
    def get_current_pair_contact_sites(
        object_1_surface_voxels,
        object_2_surface_voxels,
        contact_distance_voxels,
        overlap_voxels,
        mask=None,
    ):
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

        print_with_datetime(
            f"surface voxel coordinates,{contact_voxels_pairs.shape[1]}",
            logger,
        )
        # distances = pairwise_distances(
        #     object_1_surface_voxel_coordinates[contact_voxels_pairs],
        #     object_2_surface_voxel_coordinates[contact_voxels_pairs],
        #     "euclidean",
        # )
        # contact_voxels_pairs = np.argwhere(distances <= contact_distance_voxels)
        contact_voxels_1 = object_1_surface_voxel_coordinates[contact_voxels_pairs[0]]
        contact_voxels_2 = object_2_surface_voxel_coordinates[contact_voxels_pairs[1]]

        if len(overlap_voxels) > 0:
            indices = tuple(zip(*overlap_voxels))
            current_pair_contact_voxels[indices] = 1
        print_with_datetime("Bresenham line", logger)
        # (x_coords, y_coords, z_coords) = bresenhamline_with_mask(
        #     np.array(contact_voxels_1),
        #     np.array(contact_voxels_2),
        #     mask=None,
        #     max_iter=-1,
        # )
        # current_pair_contact_voxels[x_coords, y_coords, z_coords] = 1
        # print_with_datetime(f"Bresenham line {x_coords[:10]}", logger)
        # current_pair_contact_voxels[valid_voxels] = 1

        print_with_datetime("slower Bresenham line", logger)
        for contact_voxel_1, contact_voxel_2 in zip(contact_voxels_1, contact_voxels_2):
            valid_voxels = bresenham3DWithMask(
                *contact_voxel_1, *contact_voxel_2, mask=mask
            )
            if valid_voxels:
                x_coords, y_coords, z_coords = zip(*valid_voxels)
                current_pair_contact_voxels[x_coords, y_coords, z_coords] = 1

        current_pair_contact_sites = measure.label(
            current_pair_contact_voxels, connectivity=3
        )
        return current_pair_contact_sites.astype(np.uint64)

    @staticmethod
    def get_contact_sites(organelle_1, organelle_2):
        return

    @staticmethod
    def get_all_contact_sites_at_once(
        organelle_1,
        organelle_2,
        surface_voxels_1,
        surface_voxels_2,
        contact_boundary_1,
        contact_boundary_2,
        contact_distance_voxels,
        mask=None,
    ):
        pairwise_contact_sites = np.zeros_like(organelle_1, np.uint64)
        df = pd.DataFrame(
            columns=[
                "object_1_id",
                "object_2_id",
                "global_blockwise_contact_site",
            ]
        )
        # outputs
        print_with_datetime("Calculating pairwise contact sites", logger)
        pairwise_contact_sites = np.zeros_like(organelle_1, np.uint64)
        df = pd.DataFrame(
            columns=[
                "object_1_id",
                "object_2_id",
                "global_blockwise_contact_site",
            ]
        )
        overlap_voxels = np.argwhere((organelle_1 > 0) & (organelle_2 > 0))
        current_pair_contact_sites = ContactSites.get_current_pair_contact_sites(
            surface_voxels_1,
            surface_voxels_2,
            contact_distance_voxels,
            overlap_voxels,
            mask,
        )

        if np.any(current_pair_contact_sites):
            previous_max_id = np.amax(pairwise_contact_sites)
            current_contact_site_ids = np.unique(
                current_pair_contact_sites[current_pair_contact_sites > 0]
            )
            np.putmask(
                pairwise_contact_sites,
                current_pair_contact_sites > 0,
                current_pair_contact_sites + previous_max_id,
            )

            # Note some contact site ids may be overwritten but that shouldnt be an issue
            for current_contact_site_id in current_contact_site_ids:
                new_row = {
                    "object_1_id": 1,
                    "object_2_id": 2,
                    "global_blockwise_contact_site": current_contact_site_id
                    + previous_max_id,
                }
                df.loc[len(df)] = new_row

        return pairwise_contact_sites, df

    @staticmethod
    def get_pairwise_contact_sites(
        organelle_1,
        organelle_2,
        surface_voxels_1,
        surface_voxels_2,
        contact_boundary_1,
        contact_boundary_2,
        contact_distance_voxels,
        mask=None,
    ):

        # outputs
        print_with_datetime("Calculating pairwise contact sites", logger)
        pairwise_contact_sites = np.zeros_like(organelle_1, np.uint64)
        df = pd.DataFrame(
            columns=[
                "object_1_id",
                "object_2_id",
                "global_blockwise_contact_site",
            ]
        )

        for object_1 in np.unique(organelle_1[organelle_1 > 0]):
            for object_2 in np.unique(organelle_2[organelle_2 > 0]):
                # print_with_datetime(f"{object_1},{object_2}", logger)
                # if np.any(
                #     np.logical_and(
                #         contact_boundary_1 == object_1, contact_boundary_2 == object_2
                #     )
                # ):
                object_1_surface_voxels = surface_voxels_1 == object_1
                object_2_surface_voxels = surface_voxels_2 == object_2
                overlap_voxels = np.argwhere(
                    (organelle_1 == object_1) & (organelle_2 == object_2)
                )

                current_pair_contact_sites = (
                    ContactSites.get_current_pair_contact_sites(
                        object_1_surface_voxels,
                        object_2_surface_voxels,
                        contact_distance_voxels,
                        overlap_voxels,
                        mask,
                    )
                )

                if np.any(current_pair_contact_sites):
                    previous_max_id = np.amax(pairwise_contact_sites)
                    current_contact_site_ids = np.unique(
                        current_pair_contact_sites[current_pair_contact_sites > 0]
                    )
                    np.putmask(
                        pairwise_contact_sites,
                        current_pair_contact_sites > 0,
                        current_pair_contact_sites + previous_max_id,
                    )

                    # Note some contact site ids may be overwritten but that shouldnt be an issue
                    for current_contact_site_id in current_contact_site_ids:
                        new_row = {
                            "object_1_id": object_1,
                            "object_2_id": object_2,
                            "global_blockwise_contact_site": current_contact_site_id
                            + previous_max_id,
                        }
                        df.loc[len(df)] = new_row

        return pairwise_contact_sites, df

    def get_contact_sites(self):
        for block in self.blocks:
            self.get_blockwise_contact_sites(block)

    @staticmethod
    def get_ndarray_contact_sites_geodesic(
        organelle_1, organelle_2, contact_distance_voxels
    ):
        print_with_datetime("Surface voxels", logger)
        # erode organelles to get surface voxels
        # surface_voxels_1 = organelle_1 - erosion(organelle_1, selem=np.ones((3, 3, 3)))

        surface_voxels_1 = organelle_1 * find_boundaries(organelle_1, mode="inner")
        surface_voxels_2 = organelle_2 * find_boundaries(organelle_2, mode="inner")

        nonsurface_voxels_1 = organelle_1 - surface_voxels_1
        nonsurface_voxels_2 = organelle_2 - surface_voxels_2

        # only care about analyzing nonsurface voxel space; we want to include the surface in distance calculations
        mask = (nonsurface_voxels_2 + nonsurface_voxels_1) > 0
        print_with_datetime("Geodesic distance from organelle 1", logger)
        organelle_1_masked = ma.masked_array(surface_voxels_1 == 0, mask=mask)
        distance_from_organelle_1 = skfmm.distance(
            organelle_1_masked, dx=1, narrow=np.ceil(contact_distance_voxels)
        )
        # convert masked array values to nan
        distance_from_organelle_1 = distance_from_organelle_1.filled(np.nan)

        print_with_datetime("Geodesic distance from organelle 2", logger)
        organelle_2_masked = ma.masked_array(surface_voxels_2 == 0, mask=mask)
        distance_from_organelle_2 = skfmm.distance(
            organelle_2_masked, dx=1, narrow=np.ceil(contact_distance_voxels)
        )
        # convert masked array values to nan
        distance_from_organelle_2 = distance_from_organelle_2.filled(np.nan)
        print_with_datetime("Overlapping voxels", logger)
        overlapping_voxels = (organelle_1 > 0) & (organelle_2 > 0)
        voxels_part_of_contact_sites = (
            distance_from_organelle_1 + distance_from_organelle_2
        ) < contact_distance_voxels
        voxels_part_of_contact_sites = np.multiply(
            voxels_part_of_contact_sites, 1 - mask
        )

        voxels_part_of_contact_sites = voxels_part_of_contact_sites | overlapping_voxels

        contact_sites = measure.label(voxels_part_of_contact_sites, connectivity=3)

        return (
            # surface_voxels_1,
            # surface_voxels_2,
            # distance_from_organelle_1,
            # distance_from_organelle_2,
            organelle_1,
            organelle_2,
            contact_sites.astype(np.uint64),
        )

    @staticmethod
    def get_ndarray_contact_sites_bresenham(
        organelle_1, organelle_2, contact_distance_voxels
    ):
        print_with_datetime("Contact boundaries", logger)
        contact_boundary_1, contact_boundary_2, surface_voxels_1, surface_voxels_2 = (
            ContactSites.get_contact_boundaries(
                organelle_1, organelle_2, contact_distance_voxels
            )
        )
        # use nonsurface voxels as mask
        mask = ((organelle_1 > 0) | (organelle_2 > 0)) & (
            (surface_voxels_1 == 0) & (surface_voxels_2 == 0)
        )

        print_with_datetime("Pairwise contact sites", logger)
        contact_sites, df = ContactSites.get_all_contact_sites_at_once(
            # contact_sites, df = ContactSites.get_pairwise_contact_sites(
            organelle_1,
            organelle_2,
            surface_voxels_1,
            surface_voxels_2,
            contact_boundary_1,
            contact_boundary_2,
            contact_distance_voxels,
            mask,
        )

        return (
            organelle_1,
            organelle_2,
            # contact_boundary_1,
            # contact_boundary_2,
            # surface_voxels_1,
            # surface_voxels_2,
            contact_sites,
            # mask,
            # df,
        )

    @staticmethod
    def get_ndarray_contact_sites(
        organelle_1, organelle_2, contact_distance_voxels, method="geodesic"
    ):
        if method == "geodesic":
            return ContactSites.get_ndarray_contact_sites_geodesic(
                organelle_1, organelle_2, contact_distance_voxels
            )
        elif method == "bresenham":
            return ContactSites.get_ndarray_contact_sites_bresenham(
                organelle_1, organelle_2, contact_distance_voxels
            )

    def get_blockwise_contact_sites(self, block, method="geodesic"):
        print_with_datetime("Organelle 1 to_ndarray", logger)
        organelle_1 = self.organelle_1.to_ndarray(block.roi)
        print_with_datetime("Organelle 2 to_ndarray", logger)
        organelle_2 = self.organelle_2.to_ndarray(block.roi)

        blockwise_contact_sites = ContactSites.get_ndarray_contact_sites(
            organelle_1, organelle_2, self.contact_distance_voxels, method=method
        )
        return blockwise_contact_sites
        global_id_offset = ConnectedComponents.convertPositionToGlobalID(
            block.roi.get_begin() / self.voxel_size, organelle_1.shape
        )
        blockwise_contact_sites[blockwise_contact_sites > 0] += global_id_offset

        # write out blockwise_contact_site

    def dfer():
        df["block_id", "object_1", "object_2", "global_blockwise_contact_site"]

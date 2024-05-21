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
from cellmap_analyze.util.bresenham3d import bresenham3DWithMask
import logging
from connected_components import ConnectedComponents
import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ContactSites:
    def __init__(
        self, organelle_1_path, organelle_2_path, contact_distance_nm=10, roi=None
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

        if roi is None:
            self.roi = self.organelle_1.roi

        self.blocks = create_blocks(
            self.roi, self.organelle_1, padding=np.ceil(self.contact_distance_voxels)
        )

    def get_contact_boundaries(self, organelle_1, organelle_2):
        contact_boundary_1 = (
            expand_labels(organelle_1, distance=self.contact_distance_voxels)
            - organelle_1
        )
        contact_boundary_2 = (
            expand_labels(organelle_2, distance=self.contact_distance_voxels)
            - organelle_2
        )

        surface_voxels_1 = organelle_1[find_boundaries(organelle_1, mode="inner")]
        surface_voxels_2 = organelle_2[find_boundaries(organelle_2, mode="inner")]

        return (
            contact_boundary_1,
            contact_boundary_2,
            surface_voxels_1,
            surface_voxels_2,
        )

    @staticmethod
    def get_current_pair_contact_sites(
        object_1_surface_voxels, object_2_surface_voxels, mask=None
    ):
        current_pair_contact_voxels = np.zeros_like(object_1_surface_voxels, np.uint64)

        # get all voxel pairs that are within the contact distance
        distances = pairwise_distances(
            object_1_surface_voxels, object_2_surface_voxels, "euclidian"
        )
        contact_voxels_pairs = np.argwhere(distances <= self.contact_distance_voxels)
        contact_voxels_1 = contact_voxels_pairs[:, 0]
        contact_voxels_2 = contact_voxels_pairs[:, 1]

        for contact_voxel_1 in contact_voxels_1:
            for contact_voxel_2 in contact_voxels_2:
                valid_voxels = bresenham3DWithMask(
                    *contact_voxel_1, *contact_voxel_2, mask=mask
                )
                current_pair_contact_voxels[valid_voxels] = 1

        current_pair_contact_sites = measure.label(
            current_pair_contact_voxels, connectivity=1
        )
        return current_pair_contact_sites

    @staticmethod
    def get_pairwise_contact_sites(
        organelle_1,
        organelle_2,
        surface_voxels_1,
        surface_voxels_2,
        contact_boundary_1,
        contact_boundary_2,
        mask=None,
    ):

        # outputs
        blockwise_contact_sites = np.zeros_like(organelle_1, np.uint64)
        df = pd.DataFrame(
            columns=[
                "block_id",
                "object_1_id",
                "object_2_id",
                "global_blockwise_contact_site",
            ]
        )

        for object_1 in np.unique(organelle_1[organelle_1 > 0]):
            for object_2 in np.unique(organelle_2[organelle_2 > 0]):
                if np.any(
                    np.logical_and(
                        contact_boundary_1 == object_1, contact_boundary_2 == object_2
                    )
                ):
                    object_1_surface_voxels = np.argwhere(surface_voxels_1 == object_1)
                    object_2_surface_voxels = np.argwhere(surface_voxels_2 == object_2)

                    current_pair_contact_sites = (
                        ContactSites.get_current_pair_contact_sites(
                            object_1_surface_voxels, object_2_surface_voxels
                        )
                    )

                    if current_pair_contact_sites:
                        previous_max_id = np.amax(blockwise_contact_sites)
                        current_contact_site_ids = np.unique(
                            current_pair_contact_sites[current_pair_contact_sites > 0]
                        )
                        blockwise_contact_sites[current_pair_contact_sites > 0] = (
                            current_pair_contact_sites + previous_max_id
                        )

                        for current_contact_site_id in current_contact_site_ids:
                            new_row = {
                                "block.index": block.index,
                                "object_1_id": object_1,
                                "object_2_id": object_2,
                                "global_blockwise_contact_site": current_contact_site_id,
                            }
                            df = df.append(new_row, ignore_index=True)

    def get_blockwise_contact_sites(self, block):
        organelle_1 = self.organelle_1.to_ndarray(block.roi)
        organelle_2 = self.organelle_2.to_ndarray(block.roi)

        contact_boundary_1, contact_boundary_2, surface_voxels_1, surface_voxels_2 = (
            self.get_contact_boundaries(organelle_1, organelle_2)
        )
        mask = organelle_1 > 0 | organelle_2 > 0

        blockwise_contact_sites = self.get_pairwise_contact_sites(
            organelle_1,
            organelle_2,
            surface_voxels_1,
            surface_voxels_2,
            contact_boundary_1,
            contact_boundary_2,
            mask,
        )

        global_id_offset = ConnectedComponents.convertPositionToGlobalID(
            block.roi.get_begin() / self.voxel_size, organelle_1.shape
        )
        blockwise_contact_sites[blockwise_contact_sites > 0] += global_id_offset

        # write out blockwise_contact_site

    def dfer():
        df["block_id", "object_1", "object_2", "global_blockwise_contact_site"]

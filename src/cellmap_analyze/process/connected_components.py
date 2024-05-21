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

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ConnectedComponents:
    def __init__(self, organelle_path, roi=None):

        self.organelle = open_ds(*split_dataset_path(organelle_path))
        if roi is None:
            self.roi = self.organelle.roi

        self.blocks = create_blocks(
            self.roi, self.organelle_1, padding=np.ceil(self.contact_distance_voxels)
        )

    @staticmethod
    def convertPositionToGlobalID(position, dimensions):
        id = (
            dimensions[0] * dimensions[1] * position[2]
            + dimensions[0] * position[1]
            + position[0]
            + 1
        )
        return id

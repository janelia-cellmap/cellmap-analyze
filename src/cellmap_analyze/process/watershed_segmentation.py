# %%
import cc3d
import numpy as np
from cellmap_analyze.process.connected_components import ConnectedComponents
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
    dask_computer,
    guesstimate_npartitions,
)
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.io_util import (
    get_name_from_path,
    split_dataset_path,
)

import logging
import dask.bag as db
import numpy as np

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import edt
import fastremap

from cellmap_analyze.util.measure_util import trim_array
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.zarr_util import create_multiscale_dataset_idi

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# NOTE: Broken blockwise attempt at watershed segmentation; doesn't work because needs to be global, eg in the case of a triangle
# import numpy as np
# from scipy import ndimage as ndi
# from skimage.feature import peak_local_max
# from skimage.segmentation import watershed
# import matplotlib.pyplot as plt

# # 1. Create a blank 500×500 image
# img = np.zeros((500, 500), dtype=np.uint8)

# # 2. Define two circles of different radii
# r1, r2 = 40, 10
# center1 = (250, 150)  # (row, col)
# center2 = (250, 350)

# # 3. Draw the circles
# Y, X = np.ogrid[:500, :500]
# mask1 = (Y - center1[0])**2 + (X - center1[1])**2 <= r1**2
# mask2 = (Y - center2[0])**2 + (X - center2[1])**2 <= r2**2
# img[mask1] = 255
# img[mask2] = 255

# # 4. Draw a connecting cone (tapered bar)
# for col in range(center1[1], center2[1] + 1):
#     # interpolation factor 0→1
#     t = (col - center1[1]) / float(center2[1] - center1[1])
#     # linearly interpolated radius
#     r = r1 + t * (r2 - r1)
#     top = int(center1[0] - r)
#     bottom = int(center1[0] + r)
#     img[top:bottom + 1, col] = 255

# # 5. Compute the Euclidean distance transform
# distance = ndi.distance_transform_edt(img)

# # 6. Find the two main peaks in the distance map
# coords = peak_local_max(
#     distance,
#     footprint=np.ones((10, 10)),  # large footprint to detect two wells
#     labels=img
# )

# # 7. Create marker image from peak coordinates
# markers = np.zeros_like(img, dtype=int)
# for i, (r, c) in enumerate(coords, start=1):
#     markers[r, c] = i

# # 8. Apply watershed using these markers
# labels = watershed(-distance, markers, mask=img)

# # 9. Plot everything
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# axes[0].imshow(img, cmap="gray")
# axes[0].set_title("Original Conebell")
# axes[0].axis("off")

# axes[1].imshow(distance, cmap="magma")
# axes[1].plot(coords[:, 1], coords[:, 0], "r.", markersize=12)
# axes[1].set_title("Distance Transform + Peaks")
# axes[1].axis("off")

# axes[2].imshow(labels, cmap="nipy_spectral")
# axes[2].set_title("Watershed Segmentation")
# axes[2].axis("off")

# plt.tight_layout()
# plt.show()


class FlawedWatershedSegmentation(ComputeConfigMixin):
    def __init__(
        self,
        input_path,
        output_path=None,
        roi=None,
        delete_tmp=False,
        num_workers=10,
        pseudo_neighborhood_radius_nm=100,
    ):
        super().__init__(num_workers)
        self.input_path = input_path
        self.input_idi = ImageDataInterface(self.input_path)
        if roi is None:
            self.roi = self.input_idi.roi
        else:
            self.roi = roi

        self.voxel_size = self.input_idi.voxel_size
        self.pseudo_neighborhood_radius_voxels = int(
            np.round(pseudo_neighborhood_radius_nm / self.voxel_size[0])
        )

        output_path = output_path
        if output_path is None:
            output_path = self.input_path
            output_ds_name = get_name_from_path(output_path)
            output_ds_basepath = split_dataset_path(self.input_path)[0]
            output_path = f"{output_ds_basepath}/{output_ds_name}_watersheded"

        self.output_path = output_path

        self.distance_transform_idi = create_multiscale_dataset_idi(
            self.output_path + "_distance_transform",
            dtype=np.float32,
            voxel_size=self.voxel_size,
            total_roi=self.roi,
            write_size=self.input_idi.chunk_shape * self.voxel_size,
        )
        self.watershed_seeds_blockwise_idi = create_multiscale_dataset_idi(
            self.output_path + "_seeds_blockwise",
            dtype=np.uint64,
            voxel_size=self.voxel_size,
            total_roi=self.roi,
            write_size=self.input_idi.chunk_shape * self.voxel_size,
        )
        self.watershed_seeds_path = self.output_path + "_seeds"

        self.delete_tmp = delete_tmp

    @staticmethod
    def calculate_distance_transform_blockwise(
        block_index, input_idi, distance_transform_idi
    ):
        block = create_block_from_index(
            input_idi,
            block_index,
        )
        padding_increment_voxel = block.full_block_size[0] // (
            2 * input_idi.voxel_size[0]
        )

        max_face_value = 0
        padding_voxels = 0

        # (padding_voxels - padding_increment_voxel) is the previous padding value
        while max_face_value > (padding_voxels - padding_increment_voxel):

            padded_read_roi = block.read_roi.grow(
                padding_voxels * input_idi.voxel_size[0],
                padding_voxels * input_idi.voxel_size[0],
            )
            input = input_idi.to_ndarray_ts(padded_read_roi)
            dt = edt.edt(input, black_border=False)
            dt = trim_array(dt, padding_voxels)
            # check if we have a big enough padding
            max_face_value = np.max(
                [
                    dt[0].max(),
                    dt[-1].max(),
                    dt[:, 0].max(),
                    dt[:, -1].max(),
                    dt[:, :, 0].max(),
                    dt[:, :, -1].max(),
                ]
            )
            padding_voxels += padding_increment_voxel
        dt *= input_idi.voxel_size[0]
        input = trim_array(input, padding_voxels - padding_increment_voxel)
        distance_transform_idi.ds[block.write_roi] = dt
        return dt.max()

    def calculate_blockwise_watershed_seeds_blockwise(
        block_index,
        input_idi: ImageDataInterface,
        distance_transform_idi: ImageDataInterface,
        watershed_seeds_blockwise_idi: ImageDataInterface,
        pseudo_neighborhood_radius_voxels: int,
    ):

        block = create_block_from_index(
            input_idi,
            block_index,
            padding=input_idi.voxel_size[0] * pseudo_neighborhood_radius_voxels,
        )
        distance_transform = distance_transform_idi.to_ndarray_ts(block.read_roi)
        input = input_idi.to_ndarray_ts(block.read_roi)

        global_id_offset = block_index * np.prod(
            block.full_block_size / input_idi.voxel_size[0],
            dtype=np.uint64,
        )
        coords = peak_local_max(
            distance_transform,
            footprint=np.ones((2 * pseudo_neighborhood_radius_voxels + 1,) * 3),
            labels=input,
            exclude_border=False,
        )
        plateau_mask = np.zeros_like(distance_transform, dtype=np.uint64)
        plateau_mask[tuple(coords.T)] = 1

        plateau_mask = trim_array(plateau_mask, pseudo_neighborhood_radius_voxels)
        input = trim_array(input, pseudo_neighborhood_radius_voxels)
        plateau_mask[plateau_mask > 0] += input[plateau_mask > 0]
        plateau_labels = cc3d.connected_components(
            plateau_mask, connectivity=26, out_dtype=np.uint64
        )
        plateau_labels[plateau_labels > 0] += global_id_offset
        watershed_seeds_blockwise_idi.ds[block.write_roi] = plateau_labels

    def calculate_blockwise_watershed_seeds(self):
        num_blocks = dask_util.get_num_blocks(self.input_idi, roi=self.roi)
        block_indexes = list(range(num_blocks))
        b = db.from_sequence(
            block_indexes,
            npartitions=guesstimate_npartitions(block_indexes, self.num_workers),
        ).map(
            FlawedWatershedSegmentation.calculate_blockwise_watershed_seeds_blockwise,
            self.input_idi,
            self.distance_transform_idi,
            self.watershed_seeds_blockwise_idi,
            self.pseudo_neighborhood_radius_voxels,
        )

        with dask_util.start_dask(
            self.num_workers,
            "calculate blockwise watershed seeds",
            logger,
        ):
            with io_util.Timing_Messager(
                "Calculating blockwise watershed seeds", logger
            ):
                dask_computer(b, self.num_workers, **self.compute_args)

    def calculate_distance_transform(self):
        num_blocks = dask_util.get_num_blocks(self.input_idi, roi=self.roi)
        block_indexes = list(range(num_blocks))
        b = db.from_sequence(
            block_indexes,
            npartitions=guesstimate_npartitions(block_indexes, self.num_workers),
        ).map(
            FlawedWatershedSegmentation.calculate_distance_transform_blockwise,
            self.input_idi,
            self.distance_transform_idi,
        )
        with dask_util.start_dask(
            self.num_workers, "calculate distance transform blockwise", logger
        ):
            with io_util.Timing_Messager("Calculating distance transform", logger):
                global_dt_max = np.ceil(b.max().compute(**self.compute_args))
                self.global_dt_max_voxels = int(
                    np.ceil(global_dt_max / self.input_idi.voxel_size[0])
                )

    @staticmethod
    def watershed_blockwise(
        block_index,
        input_idi: ImageDataInterface,
        distance_transform_idi: ImageDataInterface,
        watershed_seeds_idi: ImageDataInterface,
        watershed_idi: ImageDataInterface,
        global_dt_max_voxels: int,
        pseudo_neighborhood_radius_voxels: int,
    ):
        # NOTE: Only works for uint32 or less
        padding_voxels = global_dt_max_voxels + pseudo_neighborhood_radius_voxels
        block = create_block_from_index(
            distance_transform_idi,
            block_index,
            padding=distance_transform_idi.voxel_size[0] * padding_voxels,
        )
        input = input_idi.to_ndarray_ts(block.read_roi)
        distance_transform = distance_transform_idi.to_ndarray_ts(block.read_roi)

        watershed_seeds = watershed_seeds_idi.to_ndarray_ts(block.read_roi)
        # For each seed label >0, bump its voxels that many ULPs
        # for seed_label in fastremap.unique(watershed_seeds):
        #     if seed_label <= 0:
        #         continue
        #     mask = watershed_seeds == seed_label
        #     # apply nextafter() seed_label times to those voxels
        #     for _ in range(seed_label):
        #         distance_transform[mask] = np.nextafter(
        #             distance_transform[mask],
        #             np.array(np.inf, dtype=distance_transform.dtype),
        #             dtype=distance_transform.dtype,
        #         )
        labels = np.zeros_like(distance_transform, dtype=np.uint32)
        for id in fastremap.unique(input):
            if id == 0:
                pass
            mask = input == id
            distance_transform_masked = distance_transform * mask
            watershed_seeds_masked = watershed_seeds * mask
            labels += watershed(
                -distance_transform_masked,
                markers=watershed_seeds_masked,
                mask=distance_transform_masked > 0,
                connectivity=1,
            )
        watershed_idi.ds[block.write_roi] = trim_array(labels, padding_voxels)

    def do_watershed(self):
        num_blocks = dask_util.get_num_blocks(self.input_idi, roi=self.roi)
        block_indexes = list(range(num_blocks))
        b = db.from_sequence(
            block_indexes,
            npartitions=guesstimate_npartitions(block_indexes, self.num_workers),
        ).map(
            FlawedWatershedSegmentation.watershed_blockwise,
            self.input_idi,
            self.distance_transform_idi,
            self.watershed_seeds_idi,
            self.watershed_idi,
            self.global_dt_max_voxels,
            self.pseudo_neighborhood_radius_voxels,
        )

        with dask_util.start_dask(
            self.num_workers,
            "calculate watershed blockwise",
            logger,
        ):
            with io_util.Timing_Messager("Calculating watershed", logger):
                dask_computer(b, self.num_workers, **self.compute_args)

    def get_watershed_segmentation(self):
        self.calculate_distance_transform()
        self.calculate_blockwise_watershed_seeds()

        cc = ConnectedComponents(
            connected_components_blockwise_path=self.watershed_seeds_blockwise_idi.path,
            output_path=self.watershed_seeds_path,
            object_labels_path=self.input_path,
            num_workers=self.num_workers,
            connectivity=3,
            roi=self.roi,
            delete_tmp=True,
        )
        cc.merge_connected_components_across_blocks()

        self.watershed_seeds_idi = ImageDataInterface(
            self.watershed_seeds_path + "/s0",
            mode="r+",
        )
        self.watershed_idi = create_multiscale_dataset_idi(
            self.output_path,
            dtype=self.watershed_seeds_idi.ds.dtype,
            voxel_size=self.voxel_size,
            total_roi=self.roi,
            write_size=self.input_idi.chunk_shape * self.input_idi.voxel_size,
        )
        self.do_watershed()
        if self.delete_tmp:
            dask_util.delete_tmp_dataset(
                self.watershed_seeds_idi.path,
                cc.blocks,
                self.num_workers,
                self.compute_args,
            )
            dask_util.delete_tmp_dataset(
                self.distance_transform_idi.path,
                cc.blocks,
                self.num_workers,
                self.compute_args,
            )

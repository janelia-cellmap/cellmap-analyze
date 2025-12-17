# %%
import cc3d
import numpy as np
from tqdm import tqdm
from cellmap_analyze.process.connected_components import ConnectedComponents
from cellmap_analyze.util import dask_util
from cellmap_analyze.util import io_util
from cellmap_analyze.util.dask_util import (
    create_block_from_index,
    guesstimate_npartitions,
)
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.io_util import get_output_path_from_input_path
import logging
import dask.bag as db
import numpy as np

from skimage.segmentation import watershed as skimage_watershed
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


class WatershedSegmentation(ComputeConfigMixin):
    def __init__(
        self,
        input_path,
        output_path=None,
        roi=None,
        delete_tmp=False,
        num_workers=10,
        pseudo_neighborhood_radius_nm=100,
        chunk_shape=None,
        use_deprecated_flawed=False,
    ):
        super().__init__(num_workers)
        self.input_path = input_path
        self.input_idi = ImageDataInterface(self.input_path, chunk_shape=chunk_shape)
        if roi is None:
            self.roi = self.input_idi.roi
        else:
            self.roi = roi

        self.voxel_size = self.input_idi.voxel_size
        self.pseudo_neighborhood_radius_voxels = int(
            np.round(pseudo_neighborhood_radius_nm / self.voxel_size[0])
        )

        if output_path is None:
            output_path = get_output_path_from_input_path(
                self.input_path, "_watersheded"
            )
        self.output_path = str(output_path).rstrip("/")

        # Use helper function to generate output paths (handles root datasets correctly)
        distance_transform_path = get_output_path_from_input_path(
            self.output_path, "_distance_transform"
        )
        seeds_blockwise_path = get_output_path_from_input_path(
            self.output_path, "_seeds_blockwise"
        )
        self.watershed_seeds_path = get_output_path_from_input_path(
            self.output_path, "_seeds"
        )

        self.distance_transform_idi = create_multiscale_dataset_idi(
            distance_transform_path,
            dtype=np.float32,
            voxel_size=self.voxel_size,
            total_roi=self.roi,
            write_size=self.input_idi.chunk_shape * self.voxel_size,
        )
        self.watershed_seeds_blockwise_idi = create_multiscale_dataset_idi(
            seeds_blockwise_path,
            dtype=np.uint64,
            voxel_size=self.voxel_size,
            total_roi=self.roi,
            write_size=self.input_idi.chunk_shape * self.voxel_size,
        )
        self.use_deprecated_flawed = use_deprecated_flawed
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
        dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            f"calculating blockwise watershed seeds for {self.input_idi.path}",
            WatershedSegmentation.calculate_blockwise_watershed_seeds_blockwise,
            self.input_idi,
            self.distance_transform_idi,
            self.watershed_seeds_blockwise_idi,
            self.pseudo_neighborhood_radius_voxels,
        )

    def calculate_distance_transform(self):

        num_blocks = dask_util.get_num_blocks(self.input_idi, roi=self.roi)

        # 1) Define a partition-wise helper that computes the maximum DT for its indices
        def _dt_partition(idxs, input_idi, distance_transform_idi):
            local_max = 0.0
            for idx in idxs:
                dt_val = WatershedSegmentation.calculate_distance_transform_blockwise(
                    idx, input_idi, distance_transform_idi
                )
                if dt_val > local_max:
                    local_max = dt_val
            # Return a single-element list (Dask will flatten this into one item per partition)
            return [local_max]

        # 2) Build a Bag using map_partitions instead of map
        npart = guesstimate_npartitions(num_blocks, self.num_workers)
        b = db.range(num_blocks, npartitions=npart).map_partitions(
            _dt_partition,
            self.input_idi,
            self.distance_transform_idi,
        )

        # 3) Launch Dask, compute the per-partition maxima, then take the global max
        with dask_util.start_dask(
            self.num_workers, "calculate distance transform blockwise", logger
        ):
            with io_util.TimingMessager("Calculating distance transform", logger):
                # b now contains one float per partition: the local max for that partition
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
            labels += skimage_watershed(
                -distance_transform_masked,
                markers=watershed_seeds_masked,
                mask=distance_transform_masked > 0,
                connectivity=1,
            )
        watershed_idi.ds[block.write_roi] = trim_array(labels, padding_voxels)

    def global_watershed(
        block_indexes,
        input_idi: ImageDataInterface,
        distance_transform_idi: ImageDataInterface,
        watershed_seeds_idi: ImageDataInterface,
        watershed_idi: ImageDataInterface,
    ):
        input = input_idi.to_ndarray_ts()
        distance_transform = distance_transform_idi.to_ndarray_ts()
        watershed_seeds = watershed_seeds_idi.to_ndarray_ts()
        watershed = np.zeros_like(watershed_seeds, dtype=np.uint32)
        for id in fastremap.unique(input):
            if id == 0:
                pass
            mask = input == id
            distance_transform_masked = distance_transform * mask
            watershed_seeds_masked = watershed_seeds * mask
            watershed += skimage_watershed(
                -distance_transform_masked,
                markers=watershed_seeds_masked,
                mask=distance_transform_masked > 0,
                connectivity=1,
            )
        watershed = watershed.astype(watershed_seeds.dtype)
        for block_index in tqdm(block_indexes):
            block = create_block_from_index(
                watershed_idi,
                block_index,
            )
            write_roi_voxels = block.write_roi / watershed_idi.voxel_size
            watershed_idi.ds[block.write_roi] = watershed[write_roi_voxels.to_slices()]

    def do_deprecated_flawed_watershed(self):
        num_blocks = dask_util.get_num_blocks(self.input_idi, roi=self.roi)
        dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            f"calculating deprecated and flawed blockwise watershed for {self.input_idi.path}",
            WatershedSegmentation.watershed_blockwise,
            self.input_idi,
            self.distance_transform_idi,
            self.watershed_seeds_idi,
            self.watershed_idi,
            self.global_dt_max_voxels,
            self.pseudo_neighborhood_radius_voxels,
        )

    def do_global_watershed(self):
        num_blocks = dask_util.get_num_blocks(self.input_idi, roi=self.roi)
        block_indexes = list(range(num_blocks))
        with io_util.TimingMessager("Calculating watershed", logger):
            WatershedSegmentation.global_watershed(
                block_indexes,
                self.input_idi,
                self.distance_transform_idi,
                self.watershed_seeds_idi,
                self.watershed_idi,
            )

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

        if self.use_deprecated_flawed:
            self.do_deprecated_flawed_watershed()
        else:
            self.do_global_watershed()

        if self.delete_tmp:
            dask_util.delete_tmp_dir_blockwise(
                self.watershed_seeds_idi,
                self.num_workers,
                self.compute_args,
            )
            dask_util.delete_tmp_dir_blockwise(
                self.distance_transform_idi,
                self.num_workers,
                self.compute_args,
            )

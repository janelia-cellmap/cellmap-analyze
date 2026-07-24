import logging
import os
import shutil
import uuid
from abc import ABC, abstractmethod

import edt as edt_module
import fastremap
import numpy as np
import pandas as pd
from funlib.geometry import Roi
from scipy.ndimage import binary_dilation
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from cellmap_analyze.util import dask_util, io_util
from cellmap_analyze.util.dask_util import create_block_from_index
from cellmap_analyze.util.image_data_interface import ImageDataInterface
from cellmap_analyze.util.io_util import get_output_path_from_input_path
from cellmap_analyze.util.mixins import ComputeConfigMixin
from cellmap_analyze.util.zarr_util import create_multiscale_dataset_idi

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --- Split strategies --------------------------------------------------
#
# A strategy only decides *where* to cut a single object's binary mask; all
# I/O, global ID bookkeeping, and blockwise writing lives in
# SplitNarrowBridges. See docs/split_narrow_bridges_plan.md for the design
# rationale (why this is pluggable, and the open questions around
# thresholding).


class SplitStrategy(ABC):
    @abstractmethod
    def find_subpieces(
        self, mask, voxel_size, neck_radius_voxels, minimum_subregion_volume_voxels
    ):
        """Partition a single object's binary mask into candidate subpieces.

        Args:
            mask: 3D boolean array, the object's voxels within its padded bbox.
            voxel_size: physical (original, possibly anisotropic) voxel size.
            neck_radius_voxels: candidate-cut threshold, in voxels (isotropic
                approximation -- see EDTWatershedSplit).
            minimum_subregion_volume_voxels: optional accept-gate; None
                disables it.

        Returns:
            Integer label array, same shape as ``mask``: 0 outside the mask,
            1..k inside marking k candidate subpieces. k <= 1 means "no
            split" (the caller treats this as a no-op).
        """
        raise NotImplementedError


def _merge_small_fragments(labels, min_volume_voxels):
    """Iteratively dissolve the smallest fragment below ``min_volume_voxels``
    into its largest-bordering neighbor, until every surviving fragment
    clears the gate (or only one fragment remains).

    Never drops a voxel -- a fragment with no foreground neighbor (shouldn't
    happen for a single connected object, but morphology is messy) merges
    into the overall largest fragment instead.
    """
    labels = labels.copy()
    while True:
        ids, counts = fastremap.unique(labels[labels > 0], return_counts=True)
        if len(ids) <= 1:
            break
        min_idx = np.argmin(counts)
        if counts[min_idx] >= min_volume_voxels:
            break
        smallest_id = ids[min_idx]
        dilated = binary_dilation(labels == smallest_id)
        neighbor_mask = dilated & (labels != smallest_id) & (labels > 0)
        neighbor_values = labels[neighbor_mask]
        if neighbor_values.size == 0:
            neighbor_id = ids[np.argmax(counts)]
        else:
            neighbor_ids, neighbor_counts = fastremap.unique(
                neighbor_values, return_counts=True
            )
            neighbor_id = neighbor_ids[np.argmax(neighbor_counts)]
        labels[labels == smallest_id] = neighbor_id
    return labels


class EDTWatershedSplit(SplitStrategy):
    """Distance-transform watershed: standard approach for splitting
    blob-like objects (nuclei, cells) joined by a thin neck.

    ``neck_radius_voxels`` sets the minimum separation between watershed
    seeds (an isotropic approximation using the finest voxel axis -- see the
    open question in docs/split_narrow_bridges_plan.md about anisotropic
    handling). ``minimum_subregion_volume_voxels`` is an optional post-hoc
    gate: fragments below it are dissolved into their largest neighbor
    rather than kept as separate objects.
    """

    def find_subpieces(
        self, mask, voxel_size, neck_radius_voxels, minimum_subregion_volume_voxels
    ):
        if not np.any(mask):
            return np.zeros(mask.shape, dtype=np.uint8)

        distance = edt_module.edt(mask, anisotropy=tuple(voxel_size))
        min_distance = max(1, int(round(neck_radius_voxels)))
        coords = peak_local_max(
            distance,
            min_distance=min_distance,
            labels=mask.astype(np.int32),
            exclude_border=False,
        )
        if len(coords) == 0:
            return mask.astype(np.uint8)

        markers = np.zeros(mask.shape, dtype=np.int32)
        for i, coord in enumerate(coords):
            markers[tuple(coord)] = i + 1

        labels = watershed(-distance, markers, mask=mask)

        if minimum_subregion_volume_voxels:
            labels = _merge_small_fragments(labels, minimum_subregion_volume_voxels)

        labels, _ = fastremap.renumber(labels, in_place=True)
        return labels


class SkeletonGraphSplit(SplitStrategy):
    """Skeleton/graph-based splitting for thin, branched objects (e.g.
    mitochondria) where a bare radius threshold can't distinguish a normal
    tubule from a merge artifact.

    Not implemented yet -- see docs/split_narrow_bridges_plan.md.
    """

    def find_subpieces(
        self, mask, voxel_size, neck_radius_voxels, minimum_subregion_volume_voxels
    ):
        raise NotImplementedError(
            "SkeletonGraphSplit is not implemented yet -- see "
            "docs/split_narrow_bridges_plan.md for the design. Use "
            "strategy='edt_watershed' for now."
        )


_STRATEGY_REGISTRY = {
    "edt_watershed": EDTWatershedSplit,
    "skeleton_graph": SkeletonGraphSplit,
}


class SplitNarrowBridges(ComputeConfigMixin):
    def __init__(
        self,
        segmentation_path,
        output_path,
        strategy="edt_watershed",
        neck_radius_nm=0,
        minimum_subregion_volume_nm_3=None,
        max_pieces_per_object=64,
        csv_path=None,
        num_workers=10,
        timeout=5,
        chunk_shape=None,
        delete_tmp=True,
    ):
        """
        Split accidentally-merged objects at narrow bridges, preserving all
        original foreground voxels and only changing instance labels.

        Args:
            segmentation_path: Path to the input segmentation zarr dataset.
            output_path: Path to the output segmentation dataset.
            strategy: "edt_watershed" (blob-like objects: nuclei, cells) or
                "skeleton_graph" (thin/branched objects: mitochondria; not
                yet implemented).
            neck_radius_nm: Candidate-cut threshold, in nm. Converted to
                voxels using the finest voxel axis (isotropic approximation).
            minimum_subregion_volume_nm_3: Optional accept-gate -- a
                candidate split is only kept if the resulting subregions
                (after merging any that fall below this volume into their
                largest neighbor) still number >= 2. None (default) disables
                the gate entirely.
            max_pieces_per_object: Safety cap on how many subpieces a single
                object may split into; exceeding it raises rather than
                silently truncating.
            csv_path: Optional path to a CSV with per-object bounding boxes
                (the kind Measure produces). If None, Measure is run on the
                segmentation to generate one at
                ``<output_path>/bboxes/<leaf>.csv``.
            num_workers: Number of parallel dask workers.
            timeout: Timeout for ImageDataInterface reads.
            delete_tmp: Delete the per-object scratch directory when done.
        """
        super().__init__(num_workers)
        self.segmentation_path = segmentation_path
        self.segmentation_idi = ImageDataInterface(
            segmentation_path, timeout=timeout, chunk_shape=chunk_shape
        )
        self.output_path = str(output_path).rstrip("/")
        self.roi = self.segmentation_idi.roi

        if strategy not in _STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy {strategy!r}; valid: {list(_STRATEGY_REGISTRY)}"
            )
        self.strategy_name = strategy
        self.strategy = _STRATEGY_REGISTRY[strategy]()

        if csv_path is None:
            csv_path = self._generate_bbox_csv()
        elif not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"csv_path {csv_path!r} does not exist. Pass csv_path=None "
                f"to auto-generate bboxes via Measure, or provide an "
                f"existing CSV (the kind Measure produces)."
            )
        self.csv_path = csv_path
        self.bbox_df = pd.read_csv(csv_path, index_col=0)
        self.ids = self.bbox_df.index.tolist()

        original_voxel_size = self.segmentation_idi.original_voxel_size
        voxel_volume = float(np.prod(original_voxel_size))
        self.neck_radius_voxels = float(neck_radius_nm) / min(original_voxel_size)
        if minimum_subregion_volume_nm_3 is None:
            self.minimum_subregion_volume_voxels = None
        else:
            self.minimum_subregion_volume_voxels = float(
                minimum_subregion_volume_nm_3
            ) / voxel_volume

        self.max_pieces_per_object = int(max_pieces_per_object)
        self.delete_tmp = delete_tmp
        # Per-instance suffix so concurrent runs sharing output_path don't
        # collide on the scratch/merge dirs.
        self._run_id = uuid.uuid4().hex[:8]

        logger.info(f"Loaded {len(self.ids)} IDs from {csv_path}")
        logger.info(f"Output will be written to {self.output_path}")

    def _generate_bbox_csv(self):
        """Run Measure on the segmentation to produce a per-object bbox CSV,
        the same way Skeletonize does when csv_path is not supplied."""
        from cellmap_analyze.analyze.measure import Measure
        from cellmap_analyze.util.io_util import get_leaf_name_from_path

        bbox_dir = os.path.join(self.output_path, "bboxes")
        os.makedirs(bbox_dir, exist_ok=True)
        logger.info(
            "No csv_path provided; running Measure on %s to generate "
            "per-object bboxes (output -> %s).",
            self.segmentation_path,
            bbox_dir,
        )
        Measure(
            input_path=self.segmentation_path,
            output_path=bbox_dir,
            num_workers=self.num_workers,
        ).get_measurements()

        leaf = get_leaf_name_from_path(self.segmentation_path) or "measurements"
        csv_path = os.path.join(bbox_dir, f"{leaf}.csv")
        if not os.path.exists(csv_path):
            raise RuntimeError(f"Measure did not produce expected CSV at {csv_path}")
        logger.info("Auto-generated bbox CSV at %s", csv_path)
        return csv_path

    @staticmethod
    def _object_roi(id_value, bbox_df, segmentation_idi):
        row = bbox_df.loc[id_value]
        voxel_size = segmentation_idi.voxel_size
        sf = segmentation_idi.voxel_size_scale_factor
        padding = voxel_size  # 1 voxel in each direction
        start_point = (
            np.array(
                [row["MIN Z (nm)"] * sf, row["MIN Y (nm)"] * sf, row["MIN X (nm)"] * sf]
            )
            - padding
        )
        end_point = (
            np.array(
                [row["MAX Z (nm)"] * sf, row["MAX Y (nm)"] * sf, row["MAX X (nm)"] * sf]
            )
            + padding
        )
        return Roi(start_point, end_point - start_point)

    @staticmethod
    def split_id(
        id_value,
        segmentation_idi: ImageDataInterface,
        bbox_df: pd.DataFrame,
        strategy: SplitStrategy,
        neck_radius_voxels: float,
        minimum_subregion_volume_voxels,
        max_pieces_per_object: int,
        scratch_dir: str,
    ):
        """Process a single object: extract its mask, find candidate
        subpieces, and (if split) persist the local labeling to a small
        per-object scratch file.

        Returns a dict describing the outcome; ``split_objects`` merges
        these on the driver before doing any blockwise writing.
        """
        roi = SplitNarrowBridges._object_roi(id_value, bbox_df, segmentation_idi)
        data = segmentation_idi.to_ndarray_ts(roi)
        mask = data == id_value
        if not np.any(mask):
            logger.warning(f"No voxels found for ID {id_value}, skipping")
            return {"id": int(id_value), "split": False}

        subpieces = strategy.find_subpieces(
            mask,
            segmentation_idi.original_voxel_size,
            neck_radius_voxels,
            minimum_subregion_volume_voxels,
        )
        num_pieces = int(subpieces.max()) if subpieces.size else 0
        if num_pieces <= 1:
            return {"id": int(id_value), "split": False}
        if num_pieces > max_pieces_per_object:
            raise ValueError(
                f"Object {id_value} split into {num_pieces} pieces, exceeding "
                f"max_pieces_per_object={max_pieces_per_object}. Increase the "
                f"limit or investigate why this object fragmented so heavily."
            )

        os.makedirs(scratch_dir, exist_ok=True)
        scratch_path = f"{scratch_dir}/{id_value}.npz"
        np.savez(
            scratch_path, subpieces=subpieces.astype(np.min_scalar_type(num_pieces))
        )

        return {
            "id": int(id_value),
            "split": True,
            "num_pieces": num_pieces,
            "scratch_path": scratch_path,
            "roi_begin": tuple(roi.get_begin()),
            "roi_shape": tuple(roi.shape),
        }

    @staticmethod
    def _merge_split_results(list_of_results):
        return list(list_of_results)

    @staticmethod
    def relabel_block_with_splits(
        block_index,
        segmentation_idi: ImageDataInterface,
        output_idi: ImageDataInterface,
        split_lookup: dict,
        dtype,
    ):
        block = create_block_from_index(output_idi, block_index)
        data = segmentation_idi.to_ndarray_ts(block.write_roi).astype(dtype)

        voxel_size = output_idi.voxel_size
        for id_value, (new_id_base, scratch_path, obj_roi) in split_lookup.items():
            overlap = block.write_roi.intersect(obj_roi)
            if overlap.empty:
                continue

            with np.load(scratch_path) as npz:
                subpieces = npz["subpieces"]

            data_offset = (overlap.begin - block.write_roi.begin) / voxel_size
            obj_offset = (overlap.begin - obj_roi.begin) / voxel_size
            shape = overlap.shape / voxel_size
            data_slice = tuple(
                slice(int(o), int(o + s)) for o, s in zip(data_offset, shape)
            )
            obj_slice = tuple(
                slice(int(o), int(o + s)) for o, s in zip(obj_offset, shape)
            )

            region = data[data_slice]
            sub_region = subpieces[obj_slice]
            piece_mask = (region == id_value) & (sub_region > 0)
            if np.any(piece_mask):
                region[piece_mask] = new_id_base + sub_region[piece_mask].astype(dtype)
                data[data_slice] = region

        output_idi.ds[block.write_roi] = data

    def _write_output(self, split_lookup, dtype=None):
        if dtype is None:
            dtype = self.segmentation_idi.dtype

        output_idi = create_multiscale_dataset_idi(
            self.output_path,
            dtype=dtype,
            voxel_size=self.segmentation_idi.voxel_size,
            total_roi=self.roi,
            write_size=self.segmentation_idi.chunk_shape * self.segmentation_idi.voxel_size,
            original_voxel_size=self.segmentation_idi.original_voxel_size,
        )

        num_blocks = dask_util.get_num_blocks(self.segmentation_idi, roi=self.roi)
        dask_util.compute_blockwise_partitions(
            num_blocks,
            self.num_workers,
            self.compute_args,
            logger,
            f"writing split output to {self.output_path}",
            SplitNarrowBridges.relabel_block_with_splits,
            self.segmentation_idi,
            output_idi,
            split_lookup,
            dtype,
        )

    def split_objects(self):
        """Main entry point: find candidate splits for every object, then
        write the output (a verbatim copy of the input except for split
        objects, whose voxels get new instance labels)."""
        tmp_merge_root = get_output_path_from_input_path(
            self.output_path, f"_tmp_split_metrics_to_merge_{self._run_id}"
        )
        scratch_dir = get_output_path_from_input_path(
            self.output_path, f"_split_scratch_{self._run_id}"
        )

        def _wrapper(idx):
            id_value = self.ids[idx]
            return SplitNarrowBridges.split_id(
                id_value,
                self.segmentation_idi,
                self.bbox_df,
                self.strategy,
                self.neck_radius_voxels,
                self.minimum_subregion_volume_voxels,
                self.max_pieces_per_object,
                scratch_dir,
            )

        with io_util.TimingMessager("Finding candidate splits", logger):
            results = dask_util.compute_blockwise_partitions(
                len(self.ids),
                self.num_workers,
                self.compute_args,
                logger,
                f"splitting narrow bridges for {self.segmentation_path}",
                _wrapper,
                merge_info=(SplitNarrowBridges._merge_split_results, tmp_merge_root),
            )

        split_results = [r for r in results if r["split"]]
        logger.info(f"{len(split_results)}/{len(self.ids)} objects split")

        try:
            if not split_results:
                self._write_output({})
                return

            split_results.sort(key=lambda r: r["id"])
            max_original_id = max(self.ids) if self.ids else 0
            split_lookup = {}
            for rank, r in enumerate(split_results):
                new_id_base = max_original_id + 1 + rank * self.max_pieces_per_object
                obj_roi = Roi(r["roi_begin"], r["roi_shape"])
                split_lookup[r["id"]] = (new_id_base, r["scratch_path"], obj_roi)

            max_new_id = max_original_id + len(split_results) * self.max_pieces_per_object
            new_dtype = np.min_scalar_type(max_new_id)

            self._write_output(split_lookup, new_dtype)
        finally:
            if self.delete_tmp:
                shutil.rmtree(scratch_dir, ignore_errors=True)

        logger.info("Splitting complete")

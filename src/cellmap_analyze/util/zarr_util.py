import shutil

import zarr
from cellmap_analyze.util.io_util import split_dataset_path
from cellmap_analyze.util.zarr_io import prepare_ds
import os
import shutil
from cellmap_analyze.util.image_data_interface import ImageDataInterface


# From Yuri
def create_multiscale_metadata(multsc, levels):
    # store original array in a new .zarr file as an arr_name scale
    z_attrs = multsc
    base_scale = z_attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][
        0
    ]["scale"]
    base_trans = z_attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][
        1
    ]["translation"]
    num_levels = levels
    for level in range(1, num_levels + 1):
        # break the slices up into batches, to make things easier for the dask scheduler
        sn = [dim * pow(2, level) for dim in base_scale]
        trn = [
            (dim * (pow(2, level - 1) - 0.5)) + tr
            for (dim, tr) in zip(base_scale, base_trans)
        ]

        z_attrs["multiscales"][0]["datasets"].append(
            {
                "coordinateTransformations": [
                    {"type": "scale", "scale": sn},
                    {"type": "translation", "translation": trn},
                ],
                "path": f"s{level}",
            }
        )

    return z_attrs


def generate_multiscales_metadata(
    ds_name: str,
    voxel_size: list,
    translation: list,
    units: str,
    axes: list,
):
    z_attrs: dict = {"multiscales": [{}]}
    z_attrs["multiscales"][0]["axes"] = [
        {"name": axis, "type": "space", "unit": units} for axis in axes
    ]
    z_attrs["multiscales"][0]["coordinateTransformations"] = [
        {"scale": [1.0, 1.0, 1.0], "type": "scale"}
    ]
    z_attrs["multiscales"][0]["datasets"] = [
        {
            "coordinateTransformations": [
                {"scale": voxel_size, "type": "scale"},
                {"translation": translation, "type": "translation"},
            ],
            "path": ds_name,
        }
    ]

    z_attrs["multiscales"][0]["name"] = ""
    z_attrs["multiscales"][0]["version"] = "0.4"

    return z_attrs


def write_multiscales_metadata(
    base_ds_path, ds_name, voxel_Size, translation, units, axes
):
    multiscales_metadata = generate_multiscales_metadata(
        ds_name, voxel_Size, translation, units, axes
    )
    # Write via zarr API so it works for both v2 (.zattrs) and v3 (zarr.json)
    group = zarr.open_group(base_ds_path, mode="a")
    group.attrs.update(multiscales_metadata)


def create_multiscale_dataset(
    output_path,
    dtype,
    voxel_size,
    total_roi,
    write_size,
    scale=0,
    mode="w",
    original_voxel_size=None,
):

    filename, dataset = split_dataset_path(output_path, scale=scale)
    if ("zarr" in filename or "n5" in filename) and os.path.exists(output_path):
        # open zarr store
        shutil.rmtree(output_path)

    ds = prepare_ds(
        filename=filename,
        ds_name=dataset,
        dtype=dtype,
        voxel_size=voxel_size,
        total_roi=total_roi,
        write_size=write_size,
        force_exact_write_size=True,
        multiscales_metadata=False,
        delete=mode == "w",
    )
    # For root datasets, dataset will be just "s0" (no leading slash)
    # For named datasets, dataset will be "name/s0" or just "name" if scale was added
    dataset_base = dataset.rsplit(f"/s{scale}")[0]
    if dataset_base == f"s{scale}":
        # Root dataset case: just "s0" with no leading slash
        metadata_path = filename
    else:
        # Named dataset case: has a dataset name
        metadata_path = filename + "/" + dataset_base

    # Persist the TRUE physical voxel size and offset (OME-NGFF translation) on
    # the array attrs so external readers (neuroglancer, OME tools) get correct
    # physical coordinates. The scaled integer voxel_size prepare_ds wrote is
    # internal-only; cellmap-analyze re-derives the integer scaling on read
    # (ImageDataInterface + _read_voxel_size_offset). voxel_size now holds the
    # true value (no separate original_voxel_size attr).
    from cellmap_analyze.util.voxel_size_utils import scale_voxel_size_to_integers

    effective_voxel_size = (
        list(original_voxel_size)
        if original_voxel_size is not None
        else list(voxel_size)
    )
    _, scale_factor = scale_voxel_size_to_integers(effective_voxel_size)
    # total_roi.begin is the internal funlib CORNER (= OME center - vs/2).
    # Convert corner -> OME center (+ vs/2) so the persisted translation/offset
    # is the voxel CENTER, matching OME-NGFF and what the read path expects
    # (it subtracts vs/2 back to the corner). Done for integer voxels too, so
    # there is no fall-through that leaves a raw corner on disk.
    metadata_translation = [
        float(b) / scale_factor + ev / 2.0
        for b, ev in zip(total_roi.get_begin(), effective_voxel_size)
    ]

    write_multiscales_metadata(
        metadata_path,
        f"s{scale}",
        effective_voxel_size,
        metadata_translation,
        "nanometer",
        ["z", "y", "x"],
    )

    ds.data.attrs["voxel_size"] = list(effective_voxel_size)
    ds.data.attrs["offset"] = list(metadata_translation)

    return ds


def create_multiscale_dataset_idi(
    output_path,
    dtype,
    voxel_size,
    total_roi,
    write_size,
    scale=0,
    mode="w",
    custom_fill_value=None,
    chunk_shape=None,
    original_voxel_size=None,
):
    create_multiscale_dataset(
        output_path,
        dtype,
        voxel_size,
        total_roi,
        write_size,
        scale=scale,
        mode=mode,
        original_voxel_size=original_voxel_size,
    )
    if mode == "w":
        mode = "r+"

    idi = ImageDataInterface(
        output_path + f"/s{scale}",
        mode=mode,
        custom_fill_value=custom_fill_value,
        chunk_shape=chunk_shape,
    )
    return idi

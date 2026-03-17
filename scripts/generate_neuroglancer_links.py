"""Generate neuroglancer links for each dataset with raw EM, crop affinities, and segmentations."""

import json
import os
import sys

RAW_BASE = "/nrs/cellmap/data"
PREDICTIONS_BASE = "/nrs/cellmap/ackermand/challenge/v4/predictions"
PROCESSED_BASE = "/nrs/cellmap/ackermand/challenge/v4/processed"

DATASETS = [
    "jrc_ctl-id8-1",
    "jrc_fly-vnc-1",
    "jrc_mus-kidney",
    "jrc_mus-liver-zon-2",
    "jrc_mus-nacc-1",
    "jrc_zf-cardiac-1",
]

NEUROGLANCER_BASE = "https://neuroglancer-demo.appspot.com/#!"


def find_raw_path(dataset):
    """Find the fibsem-uint8 raw data path for a dataset."""
    zarr_path = os.path.join(RAW_BASE, dataset, f"{dataset}.zarr")
    for recon in sorted(os.listdir(zarr_path)):
        candidate = os.path.join(zarr_path, recon, "em", "fibsem-uint8")
        if os.path.isdir(candidate):
            return candidate
    return None


def read_multiscales(zarr_group_path):
    """Read multiscales metadata from a zarr group .zattrs."""
    zattrs_path = os.path.join(zarr_group_path, ".zattrs")
    if not os.path.isfile(zattrs_path):
        return None
    with open(zattrs_path) as f:
        return json.load(f).get("multiscales", [{}])[0]


def local_to_http(path):
    """Convert a local filesystem path to its HTTP equivalent."""
    if path.startswith("/nrs/cellmap/"):
        return "https://cellmap-vm1.int.janelia.org/nrs/" + path[len("/nrs/cellmap/"):]
    if path.startswith("/groups/cellmap/cellmap/"):
        return "https://cellmap-vm1.int.janelia.org/prfs/" + path[len("/groups/cellmap/cellmap/"):]
    return f"file://{path}"


def make_zarr_source(path):
    """Create a neuroglancer zarr source URL from a local path."""
    return f"zarr://{local_to_http(path)}"


def build_state(dataset):
    """Build neuroglancer state JSON for a dataset."""
    layers = []

    # Raw EM layer
    raw_path = find_raw_path(dataset)
    if raw_path is None:
        print(f"  WARNING: no raw data found for {dataset}", file=sys.stderr)
        return None

    layers.append(
        {
            "type": "image",
            "source": make_zarr_source(raw_path),
            "name": f"{dataset}_raw",
            "shader": '#uicontrol invlerp normalized\nvoid main() { emitGrayscale(normalized()); }\n',
        }
    )

    # Find crops
    pred_zarr = os.path.join(PREDICTIONS_BASE, f"{dataset}.zarr")
    if not os.path.isdir(pred_zarr):
        print(f"  WARNING: no predictions zarr for {dataset}", file=sys.stderr)
        return None

    crops = sorted(
        d for d in os.listdir(pred_zarr) if os.path.isdir(os.path.join(pred_zarr, d))
    )

    for crop in crops:
        # Affinities layer — point to s0 directly and provide explicit
        # coordinate transform so neuroglancer maps dims as c', z, y, x.
        affs_path = os.path.join(pred_zarr, crop, "mito_affs")
        if os.path.isdir(affs_path):
            multiscales = read_multiscales(affs_path)
            if multiscales:
                ds0 = multiscales["datasets"][0]
                transforms = ds0["coordinateTransformations"]
                scale = transforms[0]["scale"]  # [z, y, x] in nm
                translation = transforms[1]["translation"]  # [z, y, x] in nm

                affs_source = {
                    "url": make_zarr_source(os.path.join(affs_path, "s0")),
                    "transform": {
                        "inputDimensions": {
                            "d0": [1, ""],
                            "d1": [scale[0] * 1e-9, "m"],
                            "d2": [scale[1] * 1e-9, "m"],
                            "d3": [scale[2] * 1e-9, "m"],
                        },
                        "outputDimensions": {
                            "c'": [1, ""],
                            "z": [scale[0] * 1e-9, "m"],
                            "y": [scale[1] * 1e-9, "m"],
                            "x": [scale[2] * 1e-9, "m"],
                        },
                        "matrix": [
                            [1, 0, 0, 0, 0],
                            [0, 1, 0, 0, translation[0] / scale[0] - 0.5],
                            [0, 0, 1, 0, translation[1] / scale[1] - 0.5],
                            [0, 0, 0, 1, translation[2] / scale[2] - 0.5],
                        ],
                    },
                }
            else:
                affs_source = make_zarr_source(os.path.join(affs_path, "s0"))

            layers.append(
                {
                    "type": "image",
                    "source": affs_source,
                    "name": f"{crop}_mito_affs",
                    "visible": False,
                }
            )

        # Segmentation layer
        seg_path = os.path.join(PROCESSED_BASE, f"{dataset}.zarr", crop, "mito")
        if os.path.isdir(seg_path) and os.path.isdir(os.path.join(seg_path, "s0")):
            layers.append(
                {
                    "type": "segmentation",
                    "source": make_zarr_source(seg_path),
                    "name": f"{crop}_mito_seg",
                }
            )

    state = {
        "dimensions": {"x": [1e-9, "m"], "y": [1e-9, "m"], "z": [1e-9, "m"]},
        "layers": layers,
    }
    return state


def main():
    datasets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS

    output_dir = "/nrs/cellmap/ackermand/challenge/v4"

    html_rows = []
    for dataset in datasets:
        state = build_state(dataset)
        if state is None:
            continue

        # Write state JSON file
        json_dir = os.path.join(output_dir, "neuroglancer_states")
        os.makedirs(json_dir, exist_ok=True)
        json_filename = f"neuroglancer_states/{dataset}.json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, "w") as f:
            json.dump(state, f, indent=2)

        # Build layer summary for the HTML table
        layers = state["layers"]
        crops = sorted(
            {
                layer["name"].rsplit("_mito_", 1)[0]
                for layer in layers
                if "_mito_" in layer["name"]
            }
        )
        crop_list = ", ".join(crops)

        # neuroglancer URL pointing to the JSON file via HTTP
        json_http_url = local_to_http(json_path)
        url = f"{NEUROGLANCER_BASE}{json_http_url}"

        html_rows.append((dataset, crop_list, url, json_path))
        print(f"  {dataset}: wrote {json_path}")

    # Write HTML file
    html_path = os.path.join(output_dir, "challenge_segmentations.html")
    rows_html = ""
    for dataset, crop_list, url, json_path in html_rows:
        json_http_url = local_to_http(json_path)
        rows_html += f"""\
        <tr>
          <td><a href="{url}" target="_blank">{dataset}</a></td>
          <td>{crop_list}</td>
          <td><a href="{json_http_url}" target="_blank">json</a></td>
        </tr>
"""

    html = f"""\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Challenge Segmentations — Neuroglancer</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 2rem 3rem;
      background: #f8f9fa;
      color: #202124;
    }}
    h1 {{
      font-size: 1.6rem;
      font-weight: 600;
      margin: 0 0 0.25rem 0;
    }}
    .subtitle {{
      color: #5f6368;
      font-size: 0.95rem;
      margin: 0 0 1.5rem 0;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      max-width: 800px;
      background: #fff;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    th {{
      background: #1a73e8;
      color: #fff;
      font-weight: 500;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 12px 16px;
      text-align: left;
    }}
    td {{
      padding: 12px 16px;
      border-bottom: 1px solid #e8eaed;
      font-size: 0.9rem;
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #f1f3f4; }}
    a {{
      color: #1a73e8;
      text-decoration: none;
      font-weight: 500;
    }}
    a:hover {{ text-decoration: underline; }}
    .crops {{ color: #5f6368; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <h1>Challenge Segmentations</h1>
  <p class="subtitle">
    Each link opens neuroglancer with the dataset raw EM,
    per-crop mito affinities (image layer), and mito segmentation (segmentation layer).
  </p>
  <table>
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Crops</th>
        <th>State</th>
      </tr>
    </thead>
    <tbody>
{rows_html}    </tbody>
  </table>
</body>
</html>
"""
    with open(html_path, "w") as f:
        f.write(html)
    print(f"\n  HTML: {html_path}")


if __name__ == "__main__":
    main()

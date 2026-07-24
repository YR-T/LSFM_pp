"""Estimate compact binary storage for viewer-only TrackMate positions."""

from argparse import ArgumentParser
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


DEFAULT_DATASETS = (
    "MAY05L_FOS_d",
    "MAY07L_FOS_v",
    "MAY19R_FOS_d",
    "MAY25R_FOS_v",
)


def current_path(project_root, path_string):
    path = Path(path_string)
    parts = list(path.parts)
    if "outputs" in parts:
        return project_root.joinpath(*parts[parts.index("outputs"):])
    return path


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root
        / "outputs"
        / "TrackMate_final_r2p5_q150"
        / "shared_IDXYZ_xy3"
        / "manifest.csv",
    )
    args = parser.parse_args()
    with args.manifest.open("r", encoding="utf-8", newline="") as stream:
        manifest = {row["dataset"]: row for row in csv.DictReader(stream)}

    total_spots = 0
    total_npz_bytes = 0
    with TemporaryDirectory(dir=project_root / ".tmp") as temporary:
        temporary = Path(temporary)
        for dataset in DEFAULT_DATASETS:
            xml_path = current_path(project_root, manifest[dataset]["output_xml"])
            spots = np.fromregex(
                str(xml_path),
                (
                    r'POSITION_X="([0-9eE+.\-]+)" '
                    r'POSITION_Y="([0-9eE+.\-]+)" '
                    r'POSITION_Z="([0-9eE+.\-]+)"'
                ),
                dtype=[
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                ],
            )
            # 0.1 pixel is ample for display overlays and fits 0..4095.9
            # into uint16. Z has 180 slices and fits into uint8.
            x10 = np.rint(spots["x"] * 10.0).astype(np.uint16)
            y10 = np.rint(spots["y"] * 10.0).astype(np.uint16)
            z = np.rint(spots["z"]).astype(np.uint8)
            output_path = temporary / (dataset + ".npz")
            np.savez_compressed(output_path, x10=x10, y10=y10, z=z)
            output_bytes = output_path.stat().st_size
            total_spots += len(spots)
            total_npz_bytes += output_bytes
            print(
                "{}: {:,} spots, {:.3f} bytes/spot".format(
                    dataset, len(spots), output_bytes / max(len(spots), 1)
                )
            )
    print(
        "weighted_npz_bytes_per_spot={:.6f}".format(
            total_npz_bytes / total_spots
        )
    )
    print("uncompressed_bytes_per_spot=5")


if __name__ == "__main__":
    main()

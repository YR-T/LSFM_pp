"""Validate portable viewer image archives, positions, and manifest."""

from argparse import ArgumentParser
import csv
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--viewer-root", type=Path, default=project_root / "viewer")
    args = parser.parse_args()
    with (args.viewer_root / "manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 40:
        raise RuntimeError("Expected 40 manifest rows, got {}".format(len(rows)))
    total_spots = 0
    for index, row in enumerate(rows, start=1):
        expected_shape = (int(row["height"]), int(row["width"]))
        for column in ("raw_bundle", "postprocessed_bundle"):
            path = args.viewer_root / row[column]
            with zipfile.ZipFile(path, "r") as archive:
                names = archive.namelist()
                if names != ["{:03d}.jpg".format(z) for z in range(180)]:
                    raise RuntimeError("Unexpected entries in {}".format(path))
                for z in (0, 79, 179):
                    image = np.asarray(
                        Image.open(BytesIO(archive.read("{:03d}.jpg".format(z))))
                    )
                    if image.shape != expected_shape:
                        raise RuntimeError(
                            "{} Z={} shape {} != {}".format(
                                path, z, image.shape, expected_shape
                            )
                        )
        spots_path = args.viewer_root / row["spots_bundle"]
        with np.load(spots_path) as spots:
            count = len(spots["x10"])
            if len(spots["y10"]) != count or len(spots["z"]) != count:
                raise RuntimeError("Position array mismatch: {}".format(spots_path))
            if len(spots["offsets"]) != 181:
                raise RuntimeError("Offset length mismatch: {}".format(spots_path))
            if int(spots["offsets"][-1]) != count:
                raise RuntimeError("Final offset mismatch: {}".format(spots_path))
            if int(spots["spots_after_mask"]) != count:
                raise RuntimeError("Spot count mismatch: {}".format(spots_path))
            if bool(spots["mask_applied"]) == row["dataset"].startswith("MAY08"):
                raise RuntimeError("Mask policy mismatch: {}".format(spots_path))
        total_spots += count
        print("[{}/40] {} OK".format(index, row["dataset"]))
    temporary = list(args.viewer_root.rglob("*.tmp.*"))
    if temporary:
        raise RuntimeError("Temporary files remain: {}".format(temporary))
    print("All bundles valid. Total spots: {:,}".format(total_spots))


if __name__ == "__main__":
    main()

"""Count filtered TrackMate spots retained by canonical brain priors."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import tifffile as tiff


PLACEMENT_SUFFIX = {
    "Ld": "L_FOS_d",
    "Lv": "L_FOS_v",
    "Rd": "R_FOS_d",
    "Rv": "R_FOS_v",
}


def current_path(project_root, path_string):
    path = Path(path_string)
    parts = list(path.parts)
    if "outputs" in parts:
        return project_root.joinpath(*parts[parts.index("outputs"):])
    return path


def load_csv_by_key(paths, key):
    result = {}
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                result[row[key]] = row
    return result


def placement_for_dataset(dataset):
    for placement, suffix in PLACEMENT_SUFFIX.items():
        if dataset.endswith(suffix):
            return placement
    raise ValueError("Unknown dataset placement: {}".format(dataset))


def canonical_basis(corner_row):
    angle = math.radians(float(corner_row["angle_degrees_median"]))
    wall_u = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    downward_normal = np.asarray([-math.sin(angle), math.cos(angle)], dtype=np.float64)
    inward_x = wall_u if corner_row["short_wall_side"] == "left" else -wall_u
    inward_y = (
        downward_normal
        if corner_row["long_wall_side"] == "top"
        else -downward_normal
    )
    return inward_x, inward_y


def count_one(task):
    started = time.perf_counter()
    dataset = task["dataset"]
    xml_path = Path(task["xml_path"])
    mask = tiff.imread(task["mask_path"]) > 0
    corner_row = task["corner_row"]
    inward_x, inward_y = canonical_basis(corner_row)
    corner_x = float(corner_row["corner_x_median"])
    corner_y = float(corner_row["corner_y_median"])

    spots = np.fromregex(
        str(xml_path),
        r'POSITION_X="([0-9eE+.\-]+)" POSITION_Y="([0-9eE+.\-]+)"',
        dtype=[("x", np.float32), ("y", np.float32)],
    )
    delta_x = spots["x"].astype(np.float64) - corner_x
    delta_y = spots["y"].astype(np.float64) - corner_y
    canonical_x = delta_x * inward_x[0] + delta_y * inward_x[1]
    canonical_y = delta_x * inward_y[0] + delta_y * inward_y[1]
    ix = np.rint(canonical_x).astype(np.int32)
    iy = np.rint(canonical_y).astype(np.int32)
    valid = (
        (ix >= 0)
        & (ix < mask.shape[1])
        & (iy >= 0)
        & (iy < mask.shape[0])
    )
    inside = np.zeros(len(spots), dtype=bool)
    inside[valid] = mask[iy[valid], ix[valid]]
    total = int(len(spots))
    retained = int(inside.sum())
    return {
        "dataset": dataset,
        "placement": task["placement"],
        "spots_before_mask": total,
        "spots_inside_mask": retained,
        "spots_removed": total - retained,
        "retained_fraction": retained / max(total, 1),
        "removed_fraction": (total - retained) / max(total, 1),
        "expected_spots": int(task["expected_spots"]),
        "count_matches_manifest": total == int(task["expected_spots"]),
        "input_bytes": xml_path.stat().st_size,
        "seconds": time.perf_counter() - started,
    }


def aggregate(rows, label_key):
    groups = {}
    for row in rows:
        groups.setdefault(row[label_key], []).append(row)
    output = []
    for label, items in sorted(groups.items()):
        before = sum(item["spots_before_mask"] for item in items)
        inside = sum(item["spots_inside_mask"] for item in items)
        output.append(
            {
                label_key: label,
                "datasets": len(items),
                "spots_before_mask": before,
                "spots_inside_mask": inside,
                "spots_removed": before - inside,
                "retained_fraction": inside / max(before, 1),
                "removed_fraction": (before - inside) / max(before, 1),
            }
        )
    return output


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=project_root / "outputs" / "TrackMate_final_r2p5_q150",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    mask_root = args.result_root / "brain_mask_prior_all_placements_union_final"
    output_dir = mask_root / "spot_reduction"
    output_dir.mkdir(parents=True, exist_ok=True)

    corners = load_csv_by_key(
        [
            args.result_root
            / "cuvette_corner_estimation_MAY05to08_calibrated"
            / "cuvette_corner_summary.csv",
            args.result_root
            / "cuvette_corner_estimation_MAY17plus_calibrated"
            / "cuvette_corner_summary.csv",
        ],
        "dataset",
    )
    manifest = load_csv_by_key(
        [args.result_root / "shared_IDXYZ_xy3" / "manifest.csv"], "dataset"
    )
    tasks = []
    for dataset, row in sorted(manifest.items()):
        if dataset.startswith("MAY08"):
            continue
        placement = placement_for_dataset(dataset)
        tasks.append(
            {
                "dataset": dataset,
                "placement": placement,
                "xml_path": str(current_path(project_root, row["output_xml"])),
                "mask_path": str(
                    mask_root
                    / placement
                    / (placement.lower() + "_brain_prior_mask_uint8.tif")
                ),
                "corner_row": corners[dataset],
                "expected_spots": int(row["spots"]),
            }
        )
    if len(tasks) != 36:
        raise RuntimeError("Expected 36 non-MAY08 datasets, got {}".format(len(tasks)))

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(count_one, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                "[{}/36] {} retained {:,}/{:,} ({:.1%})".format(
                    index,
                    row["dataset"],
                    row["spots_inside_mask"],
                    row["spots_before_mask"],
                    row["retained_fraction"],
                ),
                flush=True,
            )
    rows.sort(key=lambda row: row["dataset"])
    if not all(row["count_matches_manifest"] for row in rows):
        raise RuntimeError("At least one parsed XML count differs from manifest")
    by_placement = aggregate(rows, "placement")
    overall_before = sum(row["spots_before_mask"] for row in rows)
    overall_inside = sum(row["spots_inside_mask"] for row in rows)
    overall = {
        "datasets": len(rows),
        "excluded": "MAY08 (4 datasets)",
        "spots_before_mask": overall_before,
        "spots_inside_mask": overall_inside,
        "spots_removed": overall_before - overall_inside,
        "retained_fraction": overall_inside / overall_before,
        "removed_fraction": (overall_before - overall_inside) / overall_before,
        "mask_mode": "all-example union, no erosion",
    }
    write_csv(output_dir / "spot_reduction_by_dataset.csv", rows)
    write_csv(output_dir / "spot_reduction_by_placement.csv", by_placement)
    with (output_dir / "spot_reduction_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {"overall": overall, "by_placement": by_placement},
            stream,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

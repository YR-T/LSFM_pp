"""Build portable JPEG and compact-position bundles for the spot viewer."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from io import BytesIO
import json
import math
import os
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image
import tifffile as tiff


PROCESSED_SUFFIX = "_uint16_scale10000_angle_smoothed"
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
    if "data" in parts:
        return project_root.joinpath(*parts[parts.index("data"):])
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


def display_uint8(image):
    sample = image[::4, ::4].astype(np.float32)
    low, high = np.percentile(sample, [0.5, 99.8])
    scaled = (image.astype(np.float32) - low) * (255.0 / max(high - low, 1.0))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def jpeg_bytes(image, quality):
    stream = BytesIO()
    Image.fromarray(display_uint8(image), mode="L").save(
        stream, format="JPEG", quality=quality, optimize=True
    )
    return stream.getvalue()


def zip_is_complete(path, slices):
    if not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path, "r") as archive:
            return archive.testzip() is None and len(archive.namelist()) == slices
    except (OSError, zipfile.BadZipFile):
        return False


def build_raw_zip(raw_path, output_path, quality, slices, overwrite):
    if not overwrite and zip_is_complete(output_path, slices):
        return output_path.stat().st_size, True
    temporary = output_path.with_suffix(".tmp.zip")
    if temporary.exists():
        temporary.unlink()
    with tiff.TiffFile(str(raw_path)) as tif:
        if len(tif.pages) != slices:
            raise RuntimeError("{} has {} pages".format(raw_path, len(tif.pages)))
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for index, page in enumerate(tif.pages):
                archive.writestr(
                    "{:03d}.jpg".format(index),
                    jpeg_bytes(page.asarray(), quality),
                )
    os.replace(str(temporary), str(output_path))
    return output_path.stat().st_size, False


def build_processed_zip(processed_dir, output_path, quality, slices, overwrite):
    if not overwrite and zip_is_complete(output_path, slices):
        return output_path.stat().st_size, True
    paths = sorted(processed_dir.glob("*.tif"))
    if len(paths) != slices:
        raise RuntimeError("{} has {} TIFFs".format(processed_dir, len(paths)))
    temporary = output_path.with_suffix(".tmp.zip")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for index, path in enumerate(paths):
            archive.writestr(
                "{:03d}.jpg".format(index),
                jpeg_bytes(tiff.imread(str(path)), quality),
            )
    os.replace(str(temporary), str(output_path))
    return output_path.stat().st_size, False


def build_spots(task, output_path, overwrite):
    if not overwrite and output_path.is_file():
        with np.load(output_path) as data:
            return (
                output_path.stat().st_size,
                int(data["spots_after_mask"]),
                int(data["spots_before_mask"]),
                True,
            )
    spots = np.fromregex(
        task["spots_xml"],
        (
            r'POSITION_X="([0-9eE+.\-]+)" '
            r'POSITION_Y="([0-9eE+.\-]+)" '
            r'POSITION_Z="([0-9eE+.\-]+)"'
        ),
        dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)],
    )
    before = int(len(spots))
    mask_applied = task["mask_path"] is not None
    if mask_applied:
        mask = tiff.imread(task["mask_path"]) > 0
        corner_row = task["corner_row"]
        inward_x, inward_y = canonical_basis(corner_row)
        delta_x = spots["x"].astype(np.float64) - float(
            corner_row["corner_x_median"]
        )
        delta_y = spots["y"].astype(np.float64) - float(
            corner_row["corner_y_median"]
        )
        canonical_x = delta_x * inward_x[0] + delta_y * inward_x[1]
        canonical_y = delta_x * inward_y[0] + delta_y * inward_y[1]
        ix = np.rint(canonical_x).astype(np.int32)
        iy = np.rint(canonical_y).astype(np.int32)
        keep = (
            (ix >= 0)
            & (ix < mask.shape[1])
            & (iy >= 0)
            & (iy < mask.shape[0])
        )
        valid_indices = np.flatnonzero(keep)
        keep[valid_indices] &= mask[iy[valid_indices], ix[valid_indices]]
        spots = spots[keep]

    x10 = np.clip(np.rint(spots["x"] * 10.0), 0, 65535).astype(np.uint16)
    y10 = np.clip(np.rint(spots["y"] * 10.0), 0, 65535).astype(np.uint16)
    z = np.clip(np.rint(spots["z"]), 0, task["slices"] - 1).astype(np.uint8)
    order = np.argsort(z, kind="stable")
    x10 = x10[order]
    y10 = y10[order]
    z = z[order]
    offsets = np.searchsorted(
        z, np.arange(task["slices"] + 1, dtype=np.uint16), side="left"
    ).astype(np.uint32)
    temporary = output_path.with_suffix(".tmp.npz")
    if temporary.exists():
        temporary.unlink()
    np.savez_compressed(
        temporary,
        x10=x10,
        y10=y10,
        z=z,
        offsets=offsets,
        xy_scale=np.asarray(0.1, dtype=np.float32),
        spots_before_mask=np.asarray(before, dtype=np.uint32),
        spots_after_mask=np.asarray(len(z), dtype=np.uint32),
        mask_applied=np.asarray(mask_applied, dtype=np.bool_),
    )
    os.replace(str(temporary), str(output_path))
    return output_path.stat().st_size, int(len(z)), before, False


def build_dataset(task):
    started = time.perf_counter()
    viewer_root = Path(task["viewer_root"])
    dataset = task["dataset"]
    raw_path = viewer_root / "raw" / (dataset + ".zip")
    processed_path = viewer_root / "postprocessed" / (dataset + ".zip")
    spots_path = viewer_root / "spots" / (dataset + ".npz")
    raw_bytes, raw_cached = build_raw_zip(
        Path(task["raw_tif"]),
        raw_path,
        task["jpeg_quality"],
        task["slices"],
        task["overwrite"],
    )
    processed_bytes, processed_cached = build_processed_zip(
        Path(task["processed_dir"]),
        processed_path,
        task["jpeg_quality"],
        task["slices"],
        task["overwrite"],
    )
    spots_bytes, spots_after, spots_before, spots_cached = build_spots(
        task, spots_path, task["overwrite"]
    )
    return {
        "dataset": dataset,
        "placement": task["placement"],
        "raw_bundle": "raw/{}.zip".format(dataset),
        "postprocessed_bundle": "postprocessed/{}.zip".format(dataset),
        "spots_bundle": "spots/{}.npz".format(dataset),
        "width": task["width"],
        "height": task["height"],
        "slices": task["slices"],
        "jpeg_quality": task["jpeg_quality"],
        "mask_applied": not dataset.startswith("MAY08"),
        "spots_before_mask": spots_before,
        "spots_after_mask": spots_after,
        "spots_removed": spots_before - spots_after,
        "raw_bytes": raw_bytes,
        "postprocessed_bytes": processed_bytes,
        "spots_bytes": spots_bytes,
        "raw_cached": raw_cached,
        "postprocessed_cached": processed_cached,
        "spots_cached": spots_cached,
        "seconds": time.perf_counter() - started,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=project_root / "outputs" / "TrackMate_final_r2p5_q150",
    )
    parser.add_argument("--viewer-root", type=Path, default=project_root / "viewer")
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--datasets",
        help="Optional comma-separated dataset names; default builds all 40.",
    )
    args = parser.parse_args()
    for subdirectory in ("raw", "postprocessed", "spots"):
        (args.viewer_root / subdirectory).mkdir(parents=True, exist_ok=True)

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
    spot_manifest = load_csv_by_key(
        [args.result_root / "shared_IDXYZ_xy3" / "manifest.csv"], "dataset"
    )
    slice_rows = load_csv_by_key(
        [args.result_root / "robust_z_slice_summary.csv"], "dataset"
    )
    tasks = []
    for dataset, spot_row in sorted(spot_manifest.items()):
        placement = placement_for_dataset(dataset)
        processed_path = current_path(project_root, slice_rows[dataset]["input_tif"])
        processed_dir = processed_path.parent
        if not processed_dir.name.endswith(PROCESSED_SUFFIX):
            raise RuntimeError("Unexpected processed directory: {}".format(processed_dir))
        raw_stem = processed_dir.name[: -len(PROCESSED_SUFFIX)]
        raw_tif = project_root / "data" / (raw_stem + ".tif")
        with tiff.TiffFile(str(raw_tif)) as tif:
            height, width = tif.pages[0].shape
            slices = len(tif.pages)
        mask_path = None
        if not dataset.startswith("MAY08"):
            mask_path = (
                args.result_root
                / "brain_mask_prior_all_placements_union_final"
                / placement
                / (placement.lower() + "_brain_prior_mask_uint8.tif")
            )
        tasks.append(
            {
                "project_root": str(project_root),
                "viewer_root": str(args.viewer_root),
                "dataset": dataset,
                "placement": placement,
                "raw_tif": str(raw_tif),
                "processed_dir": str(processed_dir),
                "spots_xml": str(current_path(project_root, spot_row["output_xml"])),
                "corner_row": corners[dataset],
                "mask_path": str(mask_path) if mask_path is not None else None,
                "width": width,
                "height": height,
                "slices": slices,
                "jpeg_quality": args.quality,
                "overwrite": args.overwrite,
            }
        )
    if len(tasks) != 40:
        raise RuntimeError("Expected 40 datasets, got {}".format(len(tasks)))
    if args.datasets:
        selected = set(args.datasets.split(","))
        tasks = [task for task in tasks if task["dataset"] in selected]
        missing = selected - {task["dataset"] for task in tasks}
        if missing:
            raise RuntimeError("Unknown requested datasets: {}".format(sorted(missing)))

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(build_dataset, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                "[{}/{}] {} {:.2f} GB, {:,} spots, {:.1f}s".format(
                    index,
                    len(tasks),
                    result["dataset"],
                    (
                        result["raw_bytes"]
                        + result["postprocessed_bytes"]
                        + result["spots_bytes"]
                    )
                    / 1e9,
                    result["spots_after_mask"],
                    result["seconds"],
                ),
                flush=True,
            )
    results.sort(key=lambda row: row["dataset"])
    manifest_path = args.viewer_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "datasets": len(results),
        "jpeg_quality": args.quality,
        "intensity_mapping": "per-slice percentile 0.5..99.8 to uint8",
        "mask_policy": "MAY08 unmasked; all other datasets use placement union masks",
        "total_raw_bytes": sum(row["raw_bytes"] for row in results),
        "total_postprocessed_bytes": sum(
            row["postprocessed_bytes"] for row in results
        ),
        "total_spots_bytes": sum(row["spots_bytes"] for row in results),
        "total_spots": sum(row["spots_after_mask"] for row in results),
    }
    summary["total_bytes"] = (
        summary["total_raw_bytes"]
        + summary["total_postprocessed_bytes"]
        + summary["total_spots_bytes"]
    )
    with (args.viewer_root / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

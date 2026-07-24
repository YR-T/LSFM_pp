"""Estimate JPEG viewer-bundle sizes by encoding representative TIFF slices."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile as tiff


PROFILES = (
    ("full_q80", 1.0, 80),
    ("full_q85", 1.0, 85),
    ("full_q90", 1.0, 90),
    ("half_q85", 0.5, 85),
    ("half_q90", 0.5, 90),
)
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def display_uint8(image):
    sample = image[::4, ::4].astype(np.float32)
    low, high = np.percentile(sample, [0.5, 99.8])
    scaled = (image.astype(np.float32) - low) * (255.0 / max(high - low, 1.0))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def jpeg_size(image, scale, quality):
    pil_image = Image.fromarray(image, mode="L")
    if scale != 1.0:
        pil_image = pil_image.resize(
            (
                max(1, int(round(pil_image.width * scale))),
                max(1, int(round(pil_image.height * scale))),
            ),
            resample=LANCZOS,
        )
    stream = BytesIO()
    pil_image.save(stream, format="JPEG", quality=quality, optimize=True)
    return stream.tell()


def match_processed_dir(raw_path, output_root):
    candidates = sorted(
        output_root.glob(raw_path.stem + "*_uint16_scale10000_angle_smoothed")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected one processed directory for {}, got {}".format(
                raw_path.name, [path.name for path in candidates]
            )
        )
    return candidates[0]


def sample_dataset(task):
    raw_path = Path(task["raw_path"])
    processed_dir = Path(task["processed_dir"])
    processed_paths = sorted(processed_dir.glob("*.tif"))
    if len(processed_paths) != 180:
        raise RuntimeError("{} has {} processed TIFFs".format(processed_dir, len(processed_paths)))
    result = {
        "dataset": raw_path.stem,
        "raw": {name: [] for name, _, _ in PROFILES},
        "processed": {name: [] for name, _, _ in PROFILES},
    }
    with tiff.TiffFile(str(raw_path)) as tif:
        if len(tif.pages) != 180:
            raise RuntimeError("{} has {} pages".format(raw_path, len(tif.pages)))
        for index in task["indices"]:
            raw = display_uint8(tif.pages[index].asarray())
            processed = display_uint8(tiff.imread(str(processed_paths[index])))
            for name, scale, quality in PROFILES:
                result["raw"][name].append(jpeg_size(raw, scale, quality))
                result["processed"][name].append(
                    jpeg_size(processed, scale, quality)
                )
    return result


def summarize(results, images_per_collection):
    summary = {}
    for kind in ("raw", "processed"):
        summary[kind] = {}
        for profile, _, _ in PROFILES:
            sizes = np.asarray(
                [
                    size
                    for result in results
                    for size in result[kind][profile]
                ],
                dtype=np.float64,
            )
            summary[kind][profile] = {
                "sample_images": int(len(sizes)),
                "mean_bytes_per_image": float(sizes.mean()),
                "median_bytes_per_image": float(np.median(sizes)),
                "p10_bytes_per_image": float(np.percentile(sizes, 10)),
                "p90_bytes_per_image": float(np.percentile(sizes, 90)),
                "estimated_total_gb_decimal": float(
                    sizes.mean() * images_per_collection / 1e9
                ),
            }
    combined = {}
    for profile, _, _ in PROFILES:
        combined[profile] = (
            summary["raw"][profile]["estimated_total_gb_decimal"]
            + summary["processed"][profile]["estimated_total_gb_decimal"]
        )
    summary["raw_plus_processed_gb_decimal"] = combined
    return summary


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=project_root / "data")
    parser.add_argument("--output-root", type=Path, default=project_root / "outputs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--sample-indices",
        default="19,59,99,139,179",
        help="Zero-based TIFF page indices.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_root / "viewer_bundle_size_estimate.json",
    )
    args = parser.parse_args()
    indices = [int(value) for value in args.sample_indices.split(",")]
    raw_paths = sorted(args.data_root.glob("*.tif"))
    if len(raw_paths) != 40:
        raise RuntimeError("Expected 40 raw TIFFs, got {}".format(len(raw_paths)))
    tasks = [
        {
            "raw_path": str(raw_path),
            "processed_dir": str(match_processed_dir(raw_path, args.output_root)),
            "indices": indices,
        }
        for raw_path in raw_paths
    ]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(sample_dataset, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print("[{}/40] {}".format(index, result["dataset"]), flush=True)
    results.sort(key=lambda item: item["dataset"])
    report = {
        "datasets": len(raw_paths),
        "slices_per_dataset": 180,
        "images_per_collection": len(raw_paths) * 180,
        "sample_indices_zero_based": indices,
        "intensity_mapping": "per-slice percentile 0.5 to 99.8 -> uint8",
        "jpeg_profiles": [
            {"name": name, "linear_scale": scale, "quality": quality}
            for name, scale, quality in PROFILES
        ],
        "summary": summarize(results, len(raw_paths) * 180),
    }
    with args.report.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

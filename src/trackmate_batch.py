from argparse import ArgumentParser
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import re
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
from xml.sax.saxutils import quoteattr


FEATURE_BOUNDS = {
    "QUALITY": (100.0, None),
    "MIN_INTENSITY_CH1": (200.0, 700.0),
    "MEAN_INTENSITY_CH1": (200.0, 3000.0),
    "STD_INTENSITY_CH1": (300.0, 4000.0),
    "CV": (0.25, 1.2),
    "CONTRAST_CH1": (0.18, 0.7),
    "SNR_CH1": (0.7, 3.0),
    "MAX_INTENSITY_CH1": (None, None),
}

SLICE_PATTERN = re.compile(r"_(\d+)\.tif{1,2}$", re.IGNORECASE)


DetectionResult = namedtuple("DetectionResult", "image xml seconds skipped")


def parse_slice_number(path):
    match = SLICE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse slice number from {path.name}")
    return int(match.group(1))


def scijava_path(path):
    return path.resolve().as_posix()


def run_trackmate(
    image,
    raw_dir,
    fiji_exe,
    groovy_script,
    radius,
    quality_threshold,
    median_filter,
    java_memory,
    overwrite,
):
    output_xml = raw_dir / f"{image.stem}_trackmate_r3_q100_median.xml"
    if output_xml.exists() and output_xml.stat().st_size > 0 and not overwrite:
        return DetectionResult(image, output_xml, 0.0, True)

    parameters = (
        f"inputFile='{scijava_path(image)}',"
        f"outputFile='{scijava_path(output_xml)}',"
        f"radius={radius},threshold={quality_threshold},"
        f"medianFilter={'true' if median_filter else 'false'}"
    )
    command = [
        str(fiji_exe.resolve()),
        f"-Xmx{java_memory}",
        "--",
        "--headless",
        "--run",
        str(groovy_script.resolve()),
        parameters,
    ]

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=groovy_script.parent.parent,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    seconds = time.perf_counter() - started

    if completed.returncode != 0 or not output_xml.exists():
        failure_log = raw_dir.parent / "logs" / f"{image.stem}_failed.log"
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        failure_log.write_text(
            "COMMAND\n"
            + subprocess.list2cmdline(command)
            + "\n\nSTDOUT\n"
            + completed.stdout
            + "\n\nSTDERR\n"
            + completed.stderr,
            encoding="utf-8",
        )
        raise RuntimeError(
            f"TrackMate failed for {image.name} (exit {completed.returncode}); "
            f"see {failure_log}"
        )

    return DetectionResult(image, output_xml, seconds, False)


def passes_bounds(attrs):
    mean = float(attrs["MEAN_INTENSITY_CH1"])
    sd = float(attrs["STD_INTENSITY_CH1"])
    values = {
        "QUALITY": float(attrs["QUALITY"]),
        "MIN_INTENSITY_CH1": float(attrs["MIN_INTENSITY_CH1"]),
        "MEAN_INTENSITY_CH1": mean,
        "STD_INTENSITY_CH1": sd,
        "CV": sd / mean if mean != 0 else float("inf"),
        "CONTRAST_CH1": float(attrs["CONTRAST_CH1"]),
        "SNR_CH1": float(attrs["SNR_CH1"]),
        "MAX_INTENSITY_CH1": float(attrs["MAX_INTENSITY_CH1"]),
    }
    for feature, (lower, upper) in FEATURE_BOUNDS.items():
        value = values[feature]
        if lower is not None and not value > lower:
            return False
        if upper is not None and not value < upper:
            return False
    return True


def serialize_spot(attrs, indent="  "):
    serialized = " ".join(f"{key}={quoteattr(str(value))}" for key, value in attrs.items())
    return f"{indent}<Spot {serialized} />\n"


def filter_xml(
    raw_xml,
    filtered_xml,
    slice_number,
    combined_out,
    next_global_id,
):
    detected = 0
    selected = 0
    z_position = slice_number - 1

    with filtered_xml.open("w", encoding="utf-8", newline="\n") as per_slice:
        per_slice.write('<?xml version="1.0" encoding="UTF-8"?>\n<spots>\n')
        for _, element in ET.iterparse(raw_xml, events=("end",)):
            if element.tag != "Spot":
                element.clear()
                continue
            detected += 1
            attrs = dict(element.attrib)
            if passes_bounds(attrs):
                attrs["POSITION_Z"] = str(z_position)
                attrs["ID"] = str(next_global_id)
                attrs["name"] = f"ID{next_global_id}"
                line = serialize_spot(attrs)
                per_slice.write(line)
                combined_out.write(line)
                selected += 1
                next_global_id += 1
            element.clear()
        per_slice.write("</spots>\n")

    return detected, selected, next_global_id


def write_thresholds(path):
    display_names = {
        "QUALITY": "Q",
        "MIN_INTENSITY_CH1": "Min",
        "MEAN_INTENSITY_CH1": "Mean",
        "STD_INTENSITY_CH1": "SD",
        "CV": "CV",
        "CONTRAST_CH1": "Contrast",
        "SNR_CH1": "SNR",
        "MAX_INTENSITY_CH1": "Max",
    }
    payload = [
        {
            "feature": display_names[key],
            "trackmate_feature": key,
            "lower_exclusive": lower,
            "upper_exclusive": upper,
        }
        for key, (lower, upper) in FEATURE_BOUNDS.items()
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    project_root = Path(__file__).resolve().parent.parent
    default_input = (
        project_root
        / "outputs"
        / "042926_MAY08R_FOS_1_retake_c_uint16_scale10000_angle_smoothed"
    )
    default_output = project_root / "outputs" / "trackmate_042926_angle_smoothed_final"

    parser = ArgumentParser(description="Batch TrackMate LoG detection and Spot filtering.")
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument(
        "--fiji-exe",
        type=Path,
        default=Path(r"C:\Tools\Fiji.app\ImageJ-win64.exe"),
    )
    parser.add_argument(
        "--groovy-script",
        type=Path,
        default=project_root / "fiji" / "trackmate_log_detect.groovy",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--java-memory", default="12g")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    images = sorted(args.input_dir.glob("*.tif"), key=parse_slice_number)
    if len(images) != 180:
        raise RuntimeError(f"Expected 180 TIFF files, found {len(images)} in {args.input_dir}")
    if not args.fiji_exe.is_file():
        raise FileNotFoundError(args.fiji_exe)
    if not args.groovy_script.is_file():
        raise FileNotFoundError(args.groovy_script)

    raw_dir = args.output_dir / "raw_trackmate_xml"
    filtered_dir = args.output_dir / "filtered_spots_by_slice"
    raw_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    batch_started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_trackmate,
                image,
                raw_dir,
                args.fiji_exe,
                args.groovy_script,
                3.0,
                100.0,
                True,
                args.java_memory,
                args.overwrite,
            ): image
            for image in images
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            state = "cached" if result.skipped else f"{result.seconds:.1f}s"
            print(f"[{completed_count:03d}/180] {result.image.name}: {state}", flush=True)

    raw_by_image = {result.image: result.xml for result in results}
    final_xml = args.output_dir / "042926_MAY08R_FOS_1_retake_c_filtered_spots.xml"
    summary_csv = args.output_dir / "filter_summary_by_slice.csv"
    total_detected = 0
    total_selected = 0
    next_global_id = 0
    summary_rows = []

    with final_xml.open("w", encoding="utf-8", newline="\n") as combined:
        combined.write('<?xml version="1.0" encoding="UTF-8"?>\n<spots>\n')
        for image in images:
            slice_number = parse_slice_number(image)
            filtered_xml = filtered_dir / f"{image.stem}_filtered_spots.xml"
            detected, selected, next_global_id = filter_xml(
                raw_by_image[image],
                filtered_xml,
                slice_number,
                combined,
                next_global_id,
            )
            total_detected += detected
            total_selected += selected
            summary_rows.append(
                {
                    "slice": slice_number,
                    "image": image.name,
                    "detected_q_gt_100": detected,
                    "selected_all_thresholds": selected,
                }
            )
        combined.write("</spots>\n")

    with summary_csv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    write_thresholds(args.output_dir / "filter_thresholds.json")
    elapsed = time.perf_counter() - batch_started
    print(f"Raw XML files: {len(raw_by_image)}", flush=True)
    print(f"Detected Q > 100: {total_detected:,}", flush=True)
    print(f"Selected by all thresholds: {total_selected:,}", flush=True)
    print(f"Final XML: {final_xml}", flush=True)
    print(f"Elapsed: {elapsed / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

"""Run finalized LoG detection (R=2.5, median off, Q>=150) on all 7,200 slices."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET


PROCESSED_SUFFIX = "_uint16_scale10000_angle_smoothed"
DATASET_PATTERN = re.compile(
    r"^(?P<date>\d{6})_(?P<hemisphere>MAY\d+[LR])_FOS_(?P<fos>[12])(?:_|$)"
)
SLICE_PATTERN = re.compile(r"_(?P<slice>\d{3})\.tiff?$", re.IGNORECASE)
FOS_LABELS = {1: "v", 2: "d"}
RADIUS = 2.5
QUALITY_THRESHOLD = 150.0
MEDIAN_FILTER = False
EXPECTED_SLICES = 180


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def discover(outputs_dir):
    datasets = []
    seen = set()
    for directory in sorted(outputs_dir.glob("*" + PROCESSED_SUFFIX)):
        stem = directory.name[: -len(PROCESSED_SUFFIX)]
        match = DATASET_PATTERN.match(stem)
        if match is None:
            continue
        hemisphere = match.group("hemisphere")
        fos = int(match.group("fos"))
        orientation = FOS_LABELS[fos]
        dataset = "{}_FOS_{}".format(hemisphere, orientation)
        if dataset in seen:
            raise RuntimeError("Duplicate processed directory for {}".format(dataset))
        images = sorted(directory.glob("*.tif"))
        slices = []
        for image in images:
            slice_match = SLICE_PATTERN.search(image.name)
            if slice_match is None:
                raise RuntimeError("Could not parse slice from {}".format(image))
            slices.append(int(slice_match.group("slice")))
        if len(images) != EXPECTED_SLICES or slices != list(range(1, 181)):
            raise RuntimeError(
                "{}: expected slices 1..180, found {} files".format(directory, len(images))
            )
        datasets.append(
            {
                "dataset": dataset,
                "hemisphere": hemisphere,
                "orientation": orientation,
                "input_dir": directory,
                "images": images,
            }
        )
        seen.add(dataset)
    if len(datasets) != 40:
        raise RuntimeError("Expected 40 datasets, found {}".format(len(datasets)))
    hemispheres = {}
    for row in datasets:
        hemispheres.setdefault(row["hemisphere"], set()).add(row["orientation"])
    incomplete = {
        key: sorted({"d", "v"} - value)
        for key, value in hemispheres.items()
        if value != {"d", "v"}
    }
    if len(hemispheres) != 20 or incomplete:
        raise RuntimeError("Hemisphere grid incomplete: {}".format(incomplete))
    return sorted(datasets, key=lambda row: row["dataset"])


def output_name(image):
    return (
        image.stem
        + "_log_r2p5_q150_median_off.xml"
    )


def spot_count(xml_path):
    for _, element in ET.iterparse(str(xml_path), events=("start",)):
        if element.tag == "AllSpots":
            return int(element.attrib["nspots"])
    raise RuntimeError("AllSpots count not found in {}".format(xml_path))


def run_dataset(dataset, output_root, fiji_exe, groovy_script, java_memory):
    output_dir = output_root / "q150_trackmate_xml" / dataset["dataset"]
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = (
        "inputDir='{}',outputDir='{}',radius={},threshold={},"
        "medianFilter=false,overwrite=false"
    ).format(
        dataset["input_dir"].resolve().as_posix(),
        output_dir.resolve().as_posix(),
        RADIUS,
        QUALITY_THRESHOLD,
    )
    command = [
        str(fiji_exe.resolve()),
        "-Xmx{}".format(java_memory),
        "--",
        "--headless",
        "--run",
        str(groovy_script.resolve()),
        parameters,
    ]
    started = time.perf_counter()
    log_path = output_root / "logs" / (dataset["dataset"] + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write("COMMAND\n{}\n\n".format(subprocess.list2cmdline(command)))
        log.flush()
        completed = subprocess.run(
            command,
            cwd=str(groovy_script.parent.parent),
            encoding="utf-8",
            errors="replace",
            stdout=log,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            check=False,
        )
    elapsed = time.perf_counter() - started
    expected = [output_dir / output_name(image) for image in dataset["images"]]
    missing = [path for path in expected if not path.is_file() or path.stat().st_size == 0]
    if completed.returncode != 0 or missing:
        with log_path.open("a", encoding="utf-8", newline="\n") as log:
            log.write(
                "\nMISSING\n{}\n".format(
                    "\n".join(str(path) for path in missing[:20])
                )
            )
        raise RuntimeError(
            "{} failed (exit {}, missing {}); see {}".format(
                dataset["dataset"], completed.returncode, len(missing), log_path
            )
        )
    return elapsed


def build_summary(datasets, output_root):
    rows = []
    for dataset in datasets:
        output_dir = output_root / "q150_trackmate_xml" / dataset["dataset"]
        for image in dataset["images"]:
            slice_number = int(SLICE_PATTERN.search(image.name).group("slice"))
            xml_path = output_dir / output_name(image)
            if not xml_path.is_file():
                raise FileNotFoundError(xml_path)
            rows.append(
                {
                    "dataset": dataset["dataset"],
                    "hemisphere": dataset["hemisphere"],
                    "orientation": dataset["orientation"],
                    "slice": slice_number,
                    "input_tif": str(image),
                    "q150_xml": str(xml_path),
                    "spots_q150": spot_count(xml_path),
                    "xml_bytes": xml_path.stat().st_size,
                }
            )
    summary_path = output_root / "detection_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "TrackMate_final_r2p5_q150",
    )
    parser.add_argument(
        "--fiji-exe",
        type=Path,
        default=Path(r"C:\Tools\Fiji.app\ImageJ-win64.exe"),
    )
    parser.add_argument(
        "--groovy-script",
        type=Path,
        default=project_root / "fiji" / "trackmate_log_detect_directory.groovy",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--java-memory", default="8g")
    args = parser.parse_args()

    if not args.fiji_exe.is_file():
        raise FileNotFoundError(args.fiji_exe)
    if not args.groovy_script.is_file():
        raise FileNotFoundError(args.groovy_script)
    datasets = discover(project_root / "outputs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "detection_status.json"
    state = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
        "detector": "LoG",
        "radius": RADIUS,
        "quality_threshold": QUALITY_THRESHOLD,
        "median_filter": MEDIAN_FILTER,
        "dataset_total": len(datasets),
        "slice_total": len(datasets) * EXPECTED_SLICES,
        "dataset_completed": 0,
        "dataset_failed": 0,
        "workers": args.workers,
    }
    write_json(state_path, state)

    failures = []
    elapsed_by_dataset = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_dataset,
                dataset,
                args.output_dir,
                args.fiji_exe,
                args.groovy_script,
                args.java_memory,
            ): dataset
            for dataset in datasets
        }
        for index, future in enumerate(as_completed(futures), start=1):
            dataset = futures[future]
            try:
                elapsed = future.result()
                elapsed_by_dataset[dataset["dataset"]] = round(elapsed, 3)
                print(
                    "[DATASET {:02d}/40] {} complete in {:.1f} min".format(
                        index, dataset["dataset"], elapsed / 60.0
                    ),
                    flush=True,
                )
            except Exception as error:
                failures.append(
                    {
                        "dataset": dataset["dataset"],
                        "error": "{}: {}".format(type(error).__name__, error),
                    }
                )
                print("FAILED: {}".format(failures[-1]), flush=True)
            state["dataset_completed"] = len(elapsed_by_dataset)
            state["dataset_failed"] = len(failures)
            write_json(state_path, state)

    state["failures"] = failures
    if failures:
        state["status"] = "completed_with_failures"
        write_json(state_path, state)
        raise SystemExit(1)

    rows = build_summary(datasets, args.output_dir)
    state["status"] = "complete"
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["total_spots_q150"] = sum(row["spots_q150"] for row in rows)
    state["total_xml_bytes"] = sum(row["xml_bytes"] for row in rows)
    state["elapsed_by_dataset_seconds"] = elapsed_by_dataset
    write_json(state_path, state)
    print(
        "COMPLETE: {} XML, {:,} Q>=150 spots".format(
            len(rows), state["total_spots_q150"]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

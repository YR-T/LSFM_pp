"""Run a TrackMate LoG radius/median grid on representative slice 80 images."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET


SAMPLES = ("MAY06", "MAY08", "MAY18", "MAY25")
SIDES = ("L", "R")
FOS_LABELS = {1: "v", 2: "d"}
RADII = (1.5, 2.0, 2.5, 3.0)
MEDIAN_FILTERS = (False, True)
SLICE_NUMBER = 80
QUALITY_THRESHOLD = 0.0
PROCESSED_SUFFIX = "_uint16_scale10000_angle_smoothed"
DATASET_PATTERN = re.compile(
    r"^(?P<date>\d{6})_(?P<sample>MAY\d+[LR])_FOS_(?P<fos>[12])(?:_|$)"
)


def set_below_normal_priority():
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000
            )
        except Exception:
            pass


def scijava_path(path):
    return path.resolve().as_posix()


def number_tag(value):
    return "{:g}".format(value).replace(".", "p")


def discover_images(outputs_dir):
    datasets = {}
    for directory in sorted(outputs_dir.glob("*" + PROCESSED_SUFFIX)):
        if not directory.is_dir():
            continue
        stem = directory.name[: -len(PROCESSED_SUFFIX)]
        match = DATASET_PATTERN.match(stem)
        if match is None:
            continue
        sample = match.group("sample")
        sample_base = sample[:-1]
        side = sample[-1]
        fos = int(match.group("fos"))
        if sample_base not in SAMPLES or side not in SIDES:
            continue
        key = "{}_FOS_{}".format(sample, FOS_LABELS[fos])
        images = sorted(directory.glob("*_{:03d}.tif".format(SLICE_NUMBER)))
        if len(images) != 1:
            raise RuntimeError(
                "{}: expected one slice {:03d} TIFF, found {}".format(
                    directory, SLICE_NUMBER, len(images)
                )
            )
        if key in datasets:
            raise RuntimeError("Duplicate dataset for {}".format(key))
        datasets[key] = {
            "key": key,
            "sample": sample_base,
            "side": side,
            "fos": fos,
            "orientation": FOS_LABELS[fos],
            "slice": SLICE_NUMBER,
            "image": images[0],
        }

    expected = {
        "{}{}_FOS_{}".format(sample, side, FOS_LABELS[fos])
        for sample in SAMPLES
        for side in SIDES
        for fos in sorted(FOS_LABELS)
    }
    missing = sorted(expected - set(datasets))
    extra = sorted(set(datasets) - expected)
    if missing or extra:
        raise RuntimeError(
            "Dataset grid mismatch. Missing: {}; extra: {}".format(missing, extra)
        )
    return [datasets[key] for key in sorted(datasets)]


def spot_count(xml_path):
    for _, element in ET.iterparse(str(xml_path), events=("start",)):
        if element.tag == "AllSpots":
            return int(element.attrib["nspots"])
    raise RuntimeError("AllSpots count not found in {}".format(xml_path))


def run_detection(
    task,
    output_root,
    fiji_exe,
    groovy_script,
    java_memory,
    overwrite,
):
    dataset = task["dataset"]
    radius = task["radius"]
    median_filter = task["median_filter"]
    median_tag = "on" if median_filter else "off"
    filename = "{}_log_r{}_q0_median_{}.xml".format(
        dataset["image"].stem, number_tag(radius), median_tag
    )
    output_xml = output_root / "raw_xml" / dataset["key"] / filename
    output_xml.parent.mkdir(parents=True, exist_ok=True)

    if output_xml.is_file() and output_xml.stat().st_size > 0 and not overwrite:
        return {
            "dataset": dataset["key"],
            "sample": dataset["sample"],
            "side": dataset["side"],
            "fos": dataset["fos"],
            "orientation": dataset["orientation"],
            "slice": dataset["slice"],
            "detector": "LoG",
            "radius": radius,
            "median_filter": median_filter,
            "quality_threshold": QUALITY_THRESHOLD,
            "input_tif": str(dataset["image"]),
            "output_xml": str(output_xml),
            "spots": spot_count(output_xml),
            "xml_bytes": output_xml.stat().st_size,
            "seconds": 0.0,
            "cached": True,
        }

    parameters = (
        "inputFile='{}',outputFile='{}',radius={},threshold={},medianFilter={}".format(
            scijava_path(dataset["image"]),
            scijava_path(output_xml),
            radius,
            QUALITY_THRESHOLD,
            "true" if median_filter else "false",
        )
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
    completed = subprocess.run(
        command,
        cwd=str(groovy_script.parent.parent),
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    seconds = time.perf_counter() - started
    if completed.returncode != 0 or not output_xml.is_file():
        log_path = output_root / "logs" / (
            "{}_r{}_median_{}_failed.log".format(
                dataset["key"], number_tag(radius), median_tag
            )
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "COMMAND\n{}\n\nSTDOUT\n{}\n\nSTDERR\n{}".format(
                subprocess.list2cmdline(command),
                completed.stdout,
                completed.stderr,
            ),
            encoding="utf-8",
        )
        raise RuntimeError(
            "TrackMate failed for {} r={} median={}; see {}".format(
                dataset["key"], radius, median_filter, log_path
            )
        )

    return {
        "dataset": dataset["key"],
        "sample": dataset["sample"],
        "side": dataset["side"],
        "fos": dataset["fos"],
        "orientation": dataset["orientation"],
        "slice": dataset["slice"],
        "detector": "LoG",
        "radius": radius,
        "median_filter": median_filter,
        "quality_threshold": QUALITY_THRESHOLD,
        "input_tif": str(dataset["image"]),
        "output_xml": str(output_xml),
        "spots": spot_count(output_xml),
        "xml_bytes": output_xml.stat().st_size,
        "seconds": round(seconds, 3),
        "cached": False,
    }


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_summary(path, rows):
    fieldnames = [
        "dataset",
        "sample",
        "side",
        "fos",
        "orientation",
        "slice",
        "detector",
        "radius",
        "median_filter",
        "quality_threshold",
        "spots",
        "xml_bytes",
        "seconds",
        "cached",
        "input_tif",
        "output_xml",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "TMoptimization",
    )
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
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--java-memory", default="8g")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.fiji_exe.is_file():
        raise FileNotFoundError(args.fiji_exe)
    if not args.groovy_script.is_file():
        raise FileNotFoundError(args.groovy_script)
    datasets = discover_images(project_root / "outputs")
    tasks = [
        {
            "dataset": dataset,
            "radius": radius,
            "median_filter": median_filter,
        }
        for dataset in datasets
        for radius in RADII
        for median_filter in MEDIAN_FILTERS
    ]
    if len(tasks) != 128:
        raise RuntimeError("Expected 128 optimization tasks, found {}".format(len(tasks)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_below_normal_priority()
    state_path = args.output_dir / "status.json"
    state = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
        "detector": "LoG",
        "slice": SLICE_NUMBER,
        "radii": list(RADII),
        "median_filters": list(MEDIAN_FILTERS),
        "quality_threshold": QUALITY_THRESHOLD,
        "workers": args.workers,
        "total": len(tasks),
        "completed": 0,
        "failed": 0,
    }
    write_json(state_path, state)

    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_detection,
                task,
                args.output_dir,
                args.fiji_exe,
                args.groovy_script,
                args.java_memory,
                args.overwrite,
            ): task
            for task in tasks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                row = future.result()
                rows.append(row)
                cache_tag = " cached" if row["cached"] else ""
                print(
                    "[{:03d}/128] {} r={} median={} spots={:,} {:.1f}s{}".format(
                        index,
                        row["dataset"],
                        row["radius"],
                        row["median_filter"],
                        row["spots"],
                        row["seconds"],
                        cache_tag,
                    ),
                    flush=True,
                )
            except Exception as error:
                failures.append(
                    {
                        "dataset": task["dataset"]["key"],
                        "radius": task["radius"],
                        "median_filter": task["median_filter"],
                        "error": "{}: {}".format(type(error).__name__, error),
                    }
                )
                print("FAILED: {}".format(failures[-1]), flush=True)
            state["completed"] = len(rows)
            state["failed"] = len(failures)
            write_json(state_path, state)

    rows.sort(key=lambda row: (row["dataset"], row["radius"], row["median_filter"]))
    write_summary(args.output_dir / "optimization_summary.csv", rows)
    state["status"] = "complete" if not failures else "completed_with_failures"
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["failures"] = failures
    write_json(state_path, state)
    if failures:
        raise SystemExit(1)
    print(
        "COMPLETE: {} results, {:,} total detected spots".format(
            len(rows), sum(row["spots"] for row in rows)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

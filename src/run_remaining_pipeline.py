"""Run the finalized preprocessing and TrackMate pipeline for unprocessed TIFFs."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pandas as pd
import tifffile as tiff


ALREADY_PROCESSED = {
    "042926_MAY08R_FOS_1_retake_c",
    "050126_MAY08R_FOS_2_re_c",
    "062226_MAY05R_FOS_1_re2_c",
}
NAME_PATTERN = re.compile(
    r"^(?P<date>\d{6})_(?P<sample>MAY\d+[LR])_FOS_(?P<fos>[12])(?:_|$)"
)
PREPROCESS_WORKERS = 24
TRACKMATE_WORKERS = 6
TRACKMATE_JAVA_MEMORY = "8g"


def set_below_normal_priority():
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000
            )
        except Exception:
            pass


def dataset_info(input_tif, outputs_dir):
    match = NAME_PATTERN.match(input_tif.stem)
    if match is None:
        raise ValueError("Unexpected input filename: {}".format(input_tif.name))
    fos_label = "v" if match.group("fos") == "1" else "d"
    angle_csv = (
        outputs_dir
        / "preprocess_angle_series"
        / "{}_stripe_angle_estimates.csv".format(input_tif.stem)
    )
    preprocessed_dir = outputs_dir / (
        input_tif.stem + "_uint16_scale10000_angle_smoothed"
    )
    trackmate_dir = outputs_dir / "trackmate_{}_FOS_{}".format(
        match.group("sample"), fos_label
    )
    return {
        "stem": input_tif.stem,
        "input_tif": input_tif,
        "angle_csv": angle_csv,
        "preprocessed_dir": preprocessed_dir,
        "trackmate_dir": trackmate_dir,
        "final_xml": trackmate_dir / (input_tif.stem + "_filtered_spots.xml"),
    }


def discover(project_root):
    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"
    inputs = [
        path
        for path in sorted(data_dir.glob("*.tif"))
        if path.stem not in ALREADY_PROCESSED
    ]
    datasets = [dataset_info(path, outputs_dir) for path in inputs]
    targets = [str(item["trackmate_dir"]).lower() for item in datasets]
    if len(targets) != len(set(targets)):
        raise RuntimeError("TrackMate output directory collision detected.")
    return datasets


def validate_input(dataset):
    with tiff.TiffFile(str(dataset["input_tif"])) as tif_file:
        pages = len(tif_file.pages)
        shape = tuple(tif_file.pages[0].shape)
        dtype = str(tif_file.pages[0].dtype)
    if pages != 180:
        raise RuntimeError("{} has {} pages".format(dataset["stem"], pages))
    if dtype != "uint16":
        raise RuntimeError("{} has dtype {}".format(dataset["stem"], dtype))
    return {"pages": pages, "shape": shape, "dtype": dtype}


def valid_angle_csv(path):
    if not path.is_file():
        return False
    frame = pd.read_csv(str(path))
    required = {"slice", "angle_deg", "recommended_angle_deg"}
    return len(frame) == 180 and required.issubset(frame.columns)


def valid_preprocessed_dir(path):
    files = sorted(path.glob("*.tif")) if path.is_dir() else []
    manifest = path / "processing_manifest.csv"
    if len(files) != 180 or not manifest.is_file():
        return False
    frame = pd.read_csv(str(manifest))
    return len(frame) == 180 and set(frame["slice"].astype(int)) == set(range(1, 181))


def valid_trackmate_dir(dataset):
    root = dataset["trackmate_dir"]
    raw = list((root / "raw_trackmate_xml").glob("*.xml"))
    filtered = list((root / "filtered_spots_by_slice").glob("*.xml"))
    summary = root / "filter_summary_by_slice.csv"
    if (
        len(raw) != 180
        or len(filtered) != 180
        or not summary.is_file()
        or not dataset["final_xml"].is_file()
    ):
        return False
    return len(pd.read_csv(str(summary))) == 180


def write_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_command(command, log_path, project_root):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print("COMMAND: {}".format(subprocess.list2cmdline(command)), flush=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write("\nCOMMAND: {}\n".format(subprocess.list2cmdline(command)))
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            "Command failed with exit code {}. See {}".format(return_code, log_path)
        )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    project_root = Path(__file__).resolve().parent.parent
    python = Path(sys.executable).resolve()
    run_root = project_root / "outputs" / "batch_remaining_20260715_new4"
    log_dir = run_root / "logs"
    state_path = run_root / "status.json"
    run_root.mkdir(parents=True, exist_ok=True)
    set_below_normal_priority()

    datasets = [
        dataset
        for dataset in discover(project_root)
        if not valid_trackmate_dir(dataset)
    ]
    if len(datasets) != 4:
        raise RuntimeError("Expected 4 unprocessed TIFFs, found {}".format(len(datasets)))
    total = len(datasets)
    state = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "preprocess_workers": PREPROCESS_WORKERS,
        "trackmate_workers": TRACKMATE_WORKERS,
        "datasets": {},
    }
    for dataset in datasets:
        state["datasets"][dataset["stem"]] = {
            "input": str(dataset["input_tif"]),
            "preprocessed_dir": str(dataset["preprocessed_dir"]),
            "trackmate_dir": str(dataset["trackmate_dir"]),
            "status": "queued",
        }
    write_state(state_path, state)

    failures = []
    for index, dataset in enumerate(datasets, start=1):
        stem = dataset["stem"]
        record = state["datasets"][stem]
        log_path = log_dir / (stem + ".log")
        print(
            "\n===== DATASET {}/{}: {} =====".format(index, total, stem),
            flush=True,
        )
        record["status"] = "running"
        record["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_state(state_path, state)
        try:
            record["input_metadata"] = validate_input(dataset)

            if valid_angle_csv(dataset["angle_csv"]):
                print("Angle CSV already complete; skipping.", flush=True)
            else:
                run_command(
                    [
                        str(python),
                        "-u",
                        "src/estimate_stripe_angles.py",
                        str(dataset["input_tif"]),
                        str(dataset["angle_csv"]),
                        "--workers",
                        str(PREPROCESS_WORKERS),
                        "--preprocess-first",
                    ],
                    log_path,
                    project_root,
                )
            if not valid_angle_csv(dataset["angle_csv"]):
                raise RuntimeError("Angle CSV validation failed.")
            record["angle_csv"] = "complete"
            write_state(state_path, state)

            if valid_preprocessed_dir(dataset["preprocessed_dir"]):
                print("Preprocessed TIFFs already complete; skipping.", flush=True)
            else:
                run_command(
                    [
                        str(python),
                        "-u",
                        "src/preprocess_angle_batch.py",
                        str(dataset["input_tif"]),
                        str(dataset["angle_csv"]),
                        str(dataset["preprocessed_dir"]),
                        "--workers",
                        str(PREPROCESS_WORKERS),
                    ],
                    log_path,
                    project_root,
                )
            if not valid_preprocessed_dir(dataset["preprocessed_dir"]):
                raise RuntimeError("Preprocessed TIFF validation failed.")
            record["preprocessing"] = "complete"
            write_state(state_path, state)

            if valid_trackmate_dir(dataset):
                print("TrackMate outputs already complete; skipping.", flush=True)
            else:
                run_command(
                    [
                        str(python),
                        "-u",
                        "src/trackmate_batch.py",
                        "--input-dir",
                        str(dataset["preprocessed_dir"]),
                        "--output-dir",
                        str(dataset["trackmate_dir"]),
                        "--workers",
                        str(TRACKMATE_WORKERS),
                        "--java-memory",
                        TRACKMATE_JAVA_MEMORY,
                    ],
                    log_path,
                    project_root,
                )
            if not valid_trackmate_dir(dataset):
                raise RuntimeError("TrackMate output validation failed.")

            summary = pd.read_csv(
                str(dataset["trackmate_dir"] / "filter_summary_by_slice.csv")
            )
            record["detected_q_gt_100"] = int(summary["detected_q_gt_100"].sum())
            record["selected_all_thresholds"] = int(
                summary["selected_all_thresholds"].sum()
            )
            record["final_xml"] = str(dataset["final_xml"])
            record["final_xml_sha256"] = sha256(dataset["final_xml"])
            record["status"] = "complete"
            record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            write_state(state_path, state)
            print("COMPLETE: {}".format(stem), flush=True)
        except Exception as error:
            record["status"] = "failed"
            record["error"] = "{}: {}".format(type(error).__name__, error)
            record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            failures.append(stem)
            write_state(state_path, state)
            print("FAILED: {}: {}".format(stem, error), flush=True)

    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["failures"] = failures
    state["status"] = "complete" if not failures else "completed_with_failures"
    write_state(state_path, state)
    print("\nQUEUE FINISHED. failures={}".format(failures), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

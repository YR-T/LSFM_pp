"""Run hemisphere-wise final Robust Z processing in parallel and merge summaries."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

from trackmate_final_robust_z import (
    FEATURE_TRANSFORM,
    QUALITY_THRESHOLD,
    ROBUST_BOUNDS,
    ROBUST_SCALE_FACTOR,
    write_json,
)


SUMMARY_SPECS = (
    (
        "statistics",
        "robust_z_block_statistics.csv",
        [
            "block",
            "feature",
            "q150_spots",
            "transformed_median",
            "transformed_mad",
            "robust_scale",
        ],
    ),
    (
        "slices",
        "robust_z_slice_summary.csv",
        [
            "block",
            "dataset",
            "orientation",
            "slice",
            "input_tif",
            "q150_xml",
            "robust_z_xml",
            "filtered_slice_xml",
            "spots_q150",
            "spots_selected",
        ],
    ),
    (
        "conditions",
        "filtered_condition_summary.csv",
        [
            "block",
            "dataset",
            "orientation",
            "spots_q150",
            "spots_selected",
            "selected_fraction",
            "output_xml",
        ],
    ),
)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def block_complete(result_root, block):
    partial = result_root / "partial_summaries"
    state = read_json(partial / (block + "_status.json"))
    paths = [
        partial / "{}_{}.csv".format(block, suffix)
        for suffix, _, _ in SUMMARY_SPECS
    ]
    return (
        state
        and state.get("status") == "complete"
        and state.get("xml_completed") == 360
        and all(path.is_file() for path in paths)
    )


def run_block(block, result_root, project_root, python):
    if block_complete(result_root, block):
        return block, True
    log_path = result_root / "logs" / ("robust_z_" + block + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        "-u",
        "src/trackmate_final_robust_z.py",
        "--result-root",
        str(result_root),
        "--block",
        block,
    ]
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write("\nCOMMAND\n{}\n\n".format(subprocess.list2cmdline(command)))
        log.flush()
        completed = subprocess.run(
            command,
            cwd=str(project_root),
            stdout=log,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            check=False,
        )
    if completed.returncode != 0 or not block_complete(result_root, block):
        raise RuntimeError("{} failed; see {}".format(block, log_path))
    return block, False


def merge_partial_csv(result_root, blocks, suffix, output_name, fieldnames):
    rows = []
    for block in blocks:
        path = (
            result_root
            / "partial_summaries"
            / "{}_{}.csv".format(block, suffix)
        )
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    output_path = result_root / output_name
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=project_root / "outputs" / "TrackMate_final_r2p5_q150",
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    detection_summary = args.result_root / "detection_summary.csv"
    if not detection_summary.is_file():
        raise FileNotFoundError(detection_summary)
    with detection_summary.open("r", encoding="utf-8", newline="") as stream:
        detection_rows = list(csv.DictReader(stream))
    if len(detection_rows) != 7200:
        raise RuntimeError("Expected 7,200 detection rows.")
    blocks = sorted({row["hemisphere"] for row in detection_rows})
    if len(blocks) != 20:
        raise RuntimeError("Expected 20 hemisphere blocks.")

    state_path = args.result_root / "robust_z_status.json"
    state = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running_parallel",
        "quality_threshold": QUALITY_THRESHOLD,
        "feature_transform": FEATURE_TRANSFORM,
        "normalization": "median_mad",
        "robust_scale_factor": ROBUST_SCALE_FACTOR,
        "robust_bounds_exclusive": ROBUST_BOUNDS,
        "blocks_total": len(blocks),
        "blocks_completed": 0,
        "blocks_failed": 0,
        "xml_total": 7200,
        "workers": args.workers,
    }
    write_json(state_path, state)
    write_json(
        args.result_root / "filter_thresholds_robust_z.json",
        {
            "quality_threshold_inclusive": QUALITY_THRESHOLD,
            "feature_transform": FEATURE_TRANSFORM,
            "normalization": "median_mad",
            "robust_scale_factor": ROBUST_SCALE_FACTOR,
            "bounds_exclusive": ROBUST_BOUNDS,
        },
    )

    failures = []
    completed_blocks = []
    python = str(Path(sys.executable).resolve())
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_block, block, args.result_root, project_root, python
            ): block
            for block in blocks
        }
        for future in as_completed(futures):
            block = futures[future]
            try:
                _, cached = future.result()
                completed_blocks.append(block)
                print(
                    "[BLOCK {}/20] {} complete{}".format(
                        len(completed_blocks),
                        block,
                        " (cached)" if cached else "",
                    ),
                    flush=True,
                )
            except Exception as error:
                failures.append(
                    {
                        "block": block,
                        "error": "{}: {}".format(type(error).__name__, error),
                    }
                )
                print("FAILED: {}".format(failures[-1]), flush=True)
            state["blocks_completed"] = len(completed_blocks)
            state["blocks_failed"] = len(failures)
            write_json(state_path, state)
    if failures:
        state["status"] = "completed_with_failures"
        state["failures"] = failures
        write_json(state_path, state)
        raise SystemExit(1)

    merged = {}
    for suffix, output_name, fieldnames in SUMMARY_SPECS:
        merged[suffix] = merge_partial_csv(
            args.result_root, blocks, suffix, output_name, fieldnames
        )
    if len(merged["statistics"]) != 200:
        raise RuntimeError("Expected 200 statistics rows.")
    if len(merged["slices"]) != 7200:
        raise RuntimeError("Expected 7,200 slice rows.")
    if len(merged["conditions"]) != 40:
        raise RuntimeError("Expected 40 condition rows.")

    state["status"] = "complete"
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["total_q150_spots"] = sum(
        int(row["spots_q150"]) for row in merged["slices"]
    )
    state["total_selected_spots"] = sum(
        int(row["spots_selected"]) for row in merged["slices"]
    )
    state["xml_completed"] = len(merged["slices"])
    write_json(state_path, state)
    print(
        "COMPLETE: {:,} selected from {:,} Q>=150 spots".format(
            state["total_selected_spots"],
            state["total_q150_spots"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

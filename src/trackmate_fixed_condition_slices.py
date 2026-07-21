"""Run one TrackMate condition on slices 40 and 80 for every complete animal."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path

from trackmate_optimization_grid import (
    DATASET_PATTERN,
    FOS_LABELS,
    PROCESSED_SUFFIX,
    QUALITY_THRESHOLD,
    run_detection,
    set_below_normal_priority,
    write_json,
    write_summary,
)


SLICES = (40, 80)
RADIUS = 2.5
MEDIAN_FILTER = False


def discover_all_images(outputs_dir):
    datasets = {}
    conditions_by_animal = {}
    expected_conditions = {
        (side, fos) for side in ("L", "R") for fos in sorted(FOS_LABELS)
    }

    for directory in sorted(outputs_dir.glob("*" + PROCESSED_SUFFIX)):
        if not directory.is_dir():
            continue
        stem = directory.name[: -len(PROCESSED_SUFFIX)]
        match = DATASET_PATTERN.match(stem)
        if match is None:
            continue
        sample_with_side = match.group("sample")
        animal = sample_with_side[:-1]
        side = sample_with_side[-1]
        fos = int(match.group("fos"))
        key = "{}_FOS_{}".format(sample_with_side, FOS_LABELS[fos])
        if key in datasets:
            raise RuntimeError("Duplicate processed dataset for {}".format(key))

        slice_images = {}
        for slice_number in SLICES:
            images = sorted(directory.glob("*_{:03d}.tif".format(slice_number)))
            if len(images) != 1:
                raise RuntimeError(
                    "{}: expected one slice {:03d} TIFF, found {}".format(
                        directory, slice_number, len(images)
                    )
                )
            slice_images[slice_number] = images[0]

        datasets[key] = {
            "key": key,
            "sample": animal,
            "side": side,
            "fos": fos,
            "orientation": FOS_LABELS[fos],
            "slice_images": slice_images,
        }
        conditions_by_animal.setdefault(animal, set()).add((side, fos))

    incomplete = {
        animal: sorted(expected_conditions - conditions)
        for animal, conditions in conditions_by_animal.items()
        if conditions != expected_conditions
    }
    if incomplete:
        raise RuntimeError("Animals missing one or more L/R/FOS conditions: {}".format(incomplete))
    if not datasets:
        raise RuntimeError("No processed datasets found.")

    images = []
    for key in sorted(datasets):
        dataset = datasets[key]
        for slice_number in SLICES:
            images.append(
                {
                    "key": dataset["key"],
                    "sample": dataset["sample"],
                    "side": dataset["side"],
                    "fos": dataset["fos"],
                    "orientation": dataset["orientation"],
                    "slice": slice_number,
                    "image": dataset["slice_images"][slice_number],
                }
            )
    return sorted(conditions_by_animal), datasets, images


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            project_root
            / "outputs"
            / "TMoptimization"
            / "r2p5_median_off_slices_040_080"
        ),
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

    animals, datasets, images = discover_all_images(project_root / "outputs")
    tasks = [
        {
            "dataset": dataset,
            "radius": RADIUS,
            "median_filter": MEDIAN_FILTER,
        }
        for dataset in images
    ]
    expected_tasks = len(animals) * 4 * len(SLICES)
    if len(datasets) != len(animals) * 4 or len(tasks) != expected_tasks:
        raise RuntimeError(
            "Expected {} animals x 4 conditions x {} slices = {} tasks; found {}.".format(
                len(animals), len(SLICES), expected_tasks, len(tasks)
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_below_normal_priority()
    state_path = args.output_dir / "status.json"
    state = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
        "animals": animals,
        "animal_count": len(animals),
        "conditions_per_animal": 4,
        "slices": list(SLICES),
        "detector": "LoG",
        "radius": RADIUS,
        "median_filter": MEDIAN_FILTER,
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
                    "[{:03d}/{}] {} slice={:03d} spots={:,} {:.1f}s{}".format(
                        index,
                        len(tasks),
                        row["dataset"],
                        row["slice"],
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
                        "slice": task["dataset"]["slice"],
                        "error": "{}: {}".format(type(error).__name__, error),
                    }
                )
                print("FAILED: {}".format(failures[-1]), flush=True)
            state["completed"] = len(rows)
            state["failed"] = len(failures)
            write_json(state_path, state)

    rows.sort(key=lambda row: (row["dataset"], row["slice"]))
    write_summary(args.output_dir / "detection_summary.csv", rows)
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

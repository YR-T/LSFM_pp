"""Export compact sharing XML with ID and XYZ coordinates only."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from pathlib import Path
import re
import time


ATTRIBUTE_PATTERNS = {
    name: re.compile(r'\b{}="([^"]+)"'.format(name))
    for name in ("ID", "POSITION_X", "POSITION_Y", "POSITION_Z")
}


def attribute(line, name):
    match = ATTRIBUTE_PATTERNS[name].search(line)
    if match is None:
        raise ValueError("Missing {} in Spot line.".format(name))
    return match.group(1)


def compact_line(line):
    spot_id = attribute(line, "ID")
    x = float(attribute(line, "POSITION_X"))
    y = float(attribute(line, "POSITION_Y"))
    z = int(round(float(attribute(line, "POSITION_Z"))))
    return (
        '  <Spot ID="{}" POSITION_X="{:.3f}" POSITION_Y="{:.3f}" '
        'POSITION_Z="{}" />\n'
    ).format(spot_id, x, y, z)


def export_one(source, output, expected_spots, overwrite):
    if output.is_file() and output.stat().st_size > 0 and not overwrite:
        return {
            "source_xml": str(source),
            "output_xml": str(output),
            "spots": int(expected_spots),
            "output_bytes": output.stat().st_size,
            "seconds": 0.0,
            "cached": True,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    started = time.perf_counter()
    count = 0
    with source.open("r", encoding="utf-8") as input_stream:
        with temporary.open("w", encoding="utf-8", newline="\n") as output_stream:
            output_stream.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            output_stream.write("<spots>\n")
            for line in input_stream:
                if "<Spot " not in line:
                    continue
                output_stream.write(compact_line(line))
                count += 1
            output_stream.write("</spots>\n")
    if count != int(expected_spots):
        raise RuntimeError(
            "{}: expected {:,} spots, exported {:,}".format(
                source.name, int(expected_spots), count
            )
        )
    temporary.replace(output)
    return {
        "source_xml": str(source),
        "output_xml": str(output),
        "spots": count,
        "output_bytes": output.stat().st_size,
        "seconds": round(time.perf_counter() - started, 3),
        "cached": False,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=project_root / "outputs" / "TrackMate_final_r2p5_q150",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <result-root>/shared_IDXYZ_xy3.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or args.result_root / "shared_IDXYZ_xy3"
    summary_path = args.result_root / "filtered_condition_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    with summary_path.open("r", encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    if len(source_rows) != 40:
        raise RuntimeError("Expected 40 conditions, found {}".format(len(source_rows)))

    tasks = []
    for row in source_rows:
        source = Path(row["output_xml"])
        if not source.is_absolute():
            source = project_root / source
        if not source.is_file():
            raise FileNotFoundError(source)
        output = output_dir / (
            source.stem + "_IDXYZ_xy3.xml"
        )
        tasks.append((row, source, output))

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                export_one,
                source,
                output,
                int(row["spots_selected"]),
                args.overwrite,
            ): row["dataset"]
            for row, source, output in tasks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            dataset = futures[future]
            result = future.result()
            result["dataset"] = dataset
            results.append(result)
            print(
                "[{}/40] {}: {:,} spots, {:.1f} MiB, {:.1f}s{}".format(
                    index,
                    dataset,
                    result["spots"],
                    result["output_bytes"] / (1024.0 ** 2),
                    result["seconds"],
                    " cached" if result["cached"] else "",
                ),
                flush=True,
            )
    results.sort(key=lambda row: row["dataset"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = [
            "dataset",
            "spots",
            "output_bytes",
            "seconds",
            "cached",
            "source_xml",
            "output_xml",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    (output_dir / "README.txt").write_text(
        "Compact sharing XML\n"
        "Attributes: ID, POSITION_X, POSITION_Y, POSITION_Z\n"
        "POSITION_X/Y: pixel coordinates rounded to 3 decimal places\n"
        "POSITION_Z: integer slice index (0-based)\n"
        "Spot selection: final Robust Z intersection\n",
        encoding="utf-8",
    )
    print(
        "COMPLETE: {:,} spots, {:.2f} GiB".format(
            sum(row["spots"] for row in results),
            sum(row["output_bytes"] for row in results) / (1024.0 ** 3),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

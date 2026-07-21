"""Create Z-invariant 2D brain-mask QC outputs for all 40 conditions."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw


def completed(output_root, dataset, cuvette_crop, dilation_pixels, wall_margin_pixels):
    directory = output_root / dataset
    required = [
        directory / "brain_mask_2d_uint8.tif",
        directory / "mask_and_spots_qc.png",
        directory / "summary.json",
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        payload = json.loads(required[-1].read_text(encoding="utf-8"))
    except ValueError:
        return False
    cuvette = payload.get("cuvette") or {}
    return (
        payload.get("dataset") == dataset
        and payload.get("output_xml") is None
        and bool(payload.get("cuvette_crop")) == bool(cuvette_crop)
        and int(payload.get("dilation_pixels_requested", -1)) == int(dilation_pixels)
        and (
            not cuvette_crop
            or int(cuvette.get("wall_margin_pixels_effective", -1))
            == int(wall_margin_pixels)
        )
    )


def run_one(
    dataset,
    project_root,
    result_root,
    output_root,
    python,
    cuvette_crop,
    dilation_pixels,
    wall_margin_pixels,
):
    if completed(
        output_root, dataset, cuvette_crop, dilation_pixels, wall_margin_pixels
    ):
        return dataset, True
    log_path = output_root / "logs" / (dataset + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        "-u",
        "src/brain_mask_2d_prototype.py",
        "--dataset",
        dataset,
        "--result-root",
        str(result_root),
        "--output-root",
        str(output_root),
        "--mask-only",
        "--dilation-pixels",
        str(dilation_pixels),
        "--wall-margin-pixels",
        str(wall_margin_pixels),
    ]
    if cuvette_crop:
        command.append("--cuvette-crop")
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        completed_process = subprocess.run(
            command,
            cwd=str(project_root),
            stdout=log,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            check=False,
        )
    if completed_process.returncode != 0 or not completed(
        output_root, dataset, cuvette_crop, dilation_pixels, wall_margin_pixels
    ):
        raise RuntimeError("{} failed; see {}".format(dataset, log_path))
    return dataset, False


def create_contact_sheet(output_root, datasets):
    thumb_width = 480
    thumb_height = 155
    columns = 4
    rows = (len(datasets) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb_width, rows * thumb_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, dataset in enumerate(datasets):
        image = Image.open(output_root / dataset / "mask_and_spots_qc.png").convert("RGB")
        image.thumbnail((thumb_width, thumb_height - 20), Image.LANCZOS)
        x = (index % columns) * thumb_width
        y = (index // columns) * thumb_height
        canvas.paste(image, (x, y + 20))
        draw.text((x + 4, y + 2), dataset, fill="black")
    canvas.save(output_root / "all_conditions_mask_qc_contact_sheet.jpg", quality=90)


def create_qc_pdf(output_root, datasets):
    """Combine the per-condition QC PNG files into one readable PDF."""
    pages = []
    for dataset in datasets:
        with Image.open(output_root / dataset / "mask_and_spots_qc.png") as source:
            page = source.convert("RGB")
            page.thumbnail((2600, 1400), Image.LANCZOS)
            pages.append(page.copy())
    pdf_path = output_root / "all_conditions_mask_and_spots_qc.pdf"
    pages[0].save(
        pdf_path,
        "PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=150.0,
        quality=90,
    )
    for page in pages:
        page.close()
    return pdf_path


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=project_root / "outputs" / "TrackMate_final_r2p5_q150",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to <result-root>/mask_2d_all_conditions.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cuvette-crop", action="store_true")
    parser.add_argument("--dilation-pixels", type=int, default=64)
    parser.add_argument("--wall-margin-pixels", type=int, default=16)
    args = parser.parse_args()
    output_root = args.output_root or args.result_root / "mask_2d_all_conditions"
    output_root.mkdir(parents=True, exist_ok=True)
    with (args.result_root / "robust_z_slice_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        datasets = sorted({row["dataset"] for row in csv.DictReader(stream)})
    if len(datasets) != 40:
        raise RuntimeError("Expected 40 datasets, found {}".format(len(datasets)))

    python = str(Path(sys.executable).resolve())
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one,
                dataset,
                project_root,
                args.result_root,
                output_root,
                python,
                args.cuvette_crop,
                args.dilation_pixels,
                args.wall_margin_pixels,
            ): dataset
            for dataset in datasets
        }
        for index, future in enumerate(as_completed(futures), start=1):
            dataset = futures[future]
            try:
                _, cached = future.result()
                print(
                    "[{}/40] {} complete{}".format(
                        index, dataset, " (cached)" if cached else ""
                    ),
                    flush=True,
                )
            except Exception as error:
                failures.append(
                    {"dataset": dataset, "error": "{}: {}".format(type(error).__name__, error)}
                )
                print("FAILED: {}".format(failures[-1]), flush=True)
    if failures:
        (output_root / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        raise SystemExit(1)

    summaries = []
    for dataset in datasets:
        payload = json.loads(
            (output_root / dataset / "summary.json").read_text(encoding="utf-8")
        )
        summaries.append(payload)
    fields = [
        "dataset",
        "mask_fraction",
        "spots_before",
        "spots_kept",
        "spots_removed",
        "removed_fraction",
        "projection_otsu_threshold",
        "dilation_pixels_effective",
        "elapsed_seconds",
    ]
    with (output_root / "mask_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    create_contact_sheet(output_root, datasets)
    pdf_path = create_qc_pdf(output_root, datasets)
    print("COMPLETE: 40 mask/QC outputs", flush=True)
    print("QC PDF: {}".format(pdf_path), flush=True)


if __name__ == "__main__":
    main()

"""Parallel angle-aware preprocessing for a multi-page TIFF stack."""

import os

# Avoid nested numerical-library threading inside process workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pystripe
from scipy.ndimage import rotate
import tifffile as tiff

try:
    from preprocess import flatfield_like_correction, subtract_background_morphology
except ImportError:
    from .preprocess import flatfield_like_correction, subtract_background_morphology


def _set_below_normal_priority():
    """Let interactive desktop applications pre-empt batch workers on Windows."""
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000
            )
        except Exception:
            pass


def _preprocess_at_angle(raw, correction_angle, pad=128):
    raw = np.asarray(raw, dtype=np.float32)
    restore_transpose = raw.shape[1] > raw.shape[0]
    if restore_transpose:
        raw = raw.T
    offset = float(np.percentile(raw, 0.05))
    raw -= offset
    raw[raw < 0] = 0

    padded = np.pad(raw, ((pad, pad), (pad, pad)), mode="reflect")
    aligned = rotate(
        padded,
        angle=float(correction_angle),
        reshape=False,
        order=1,
        mode="reflect",
        prefilter=False,
    )
    aligned = pystripe.filter_streaks(
        aligned, sigma=(128, 256), level=7, wavelet="db2"
    ).astype(np.float32)
    restored = rotate(
        aligned,
        angle=-float(correction_angle),
        reshape=False,
        order=1,
        mode="reflect",
        prefilter=False,
    )[pad:-pad, pad:-pad]

    ffc, _ = flatfield_like_correction(
        restored, sigma=120, reference_level=100, max_gain=3.0
    )
    preprocessed, _ = subtract_background_morphology(ffc, radius=20)
    if restore_transpose:
        preprocessed = preprocessed.T
    return preprocessed, offset


def _process_chunk(args):
    input_tif, output_dir, input_stem, jobs, output_scale = args
    _set_below_normal_priority()
    records = []
    with tiff.TiffFile(str(input_tif)) as tif:
        for page_index, estimated_angle, applied_angle in jobs:
            started = time.time()
            output_path = output_dir / "{}_{:03d}.tif".format(
                input_stem, page_index + 1
            )
            preprocessed, offset = _preprocess_at_angle(
                tif.pages[page_index].asarray(), applied_angle
            )
            output_uint16 = np.clip(
                preprocessed * output_scale, 0, 65535
            ).astype(np.uint16)
            tiff.imwrite(str(output_path), output_uint16)
            records.append(
                {
                    "slice": page_index + 1,
                    "output": output_path.name,
                    "estimated_angle_deg": estimated_angle,
                    "applied_angle_deg": applied_angle,
                    "offset": offset,
                    "seconds": time.time() - started,
                }
            )
    return records


def run_batch(
    input_tif,
    angle_csv,
    output_dir,
    workers=12,
    overwrite=False,
    output_scale=100.0,
):
    input_tif = Path(input_tif)
    angle_csv = Path(angle_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "processing_manifest.csv"

    angles = pd.read_csv(str(angle_csv))
    required = {"slice", "angle_deg", "recommended_angle_deg"}
    if not required.issubset(angles.columns):
        raise ValueError("Angle CSV is missing: {}".format(sorted(required - set(angles.columns))))

    with tiff.TiffFile(str(input_tif)) as tif:
        page_count = len(tif.pages)
    if page_count != len(angles):
        raise ValueError(
            "TIFF has {} pages but angle CSV has {} rows".format(page_count, len(angles))
        )

    jobs = []
    skipped = 0
    for page_index in range(page_count):
        output_path = output_dir / "{}_{:03d}.tif".format(
            input_tif.stem, page_index + 1
        )
        if output_path.exists() and not overwrite:
            skipped += 1
            continue
        row = angles.iloc[page_index]
        jobs.append(
            (page_index, float(row["angle_deg"]), float(row["recommended_angle_deg"]))
        )

    if not jobs:
        print("Nothing to process; all {} pages already exist.".format(page_count))
        return []

    workers = max(1, min(int(workers), len(jobs)))
    # Contiguous chunks reduce random reads from the multi-page TIFF.
    chunk_size = (len(jobs) + workers - 1) // workers
    chunks = [jobs[i : i + chunk_size] for i in range(0, len(jobs), chunk_size)]
    args = [
        (input_tif, output_dir, input_tif.stem, chunk, float(output_scale))
        for chunk in chunks
    ]

    print(
        "Starting {} pages with {} below-normal-priority workers ({} skipped).".format(
            len(jobs), workers, skipped
        ),
        flush=True,
    )
    started = time.time()
    records = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_chunk, item) for item in args]
        for future in as_completed(futures):
            chunk_records = future.result()
            records.extend(chunk_records)
            print(
                "Completed {}/{} pages ({:.1f} min)".format(
                    len(records), len(jobs), (time.time() - started) / 60.0
                ),
                flush=True,
            )

    new_manifest = pd.DataFrame(records)
    if manifest_path.exists() and not overwrite:
        old_manifest = pd.read_csv(str(manifest_path))
        new_manifest = pd.concat([old_manifest, new_manifest], ignore_index=True)
    new_manifest = new_manifest.drop_duplicates("slice", keep="last").sort_values("slice")
    new_manifest.to_csv(str(manifest_path), index=False)
    print(
        "Done: saved={}, skipped={}, elapsed={:.1f} min".format(
            len(records), skipped, (time.time() - started) / 60.0
        ),
        flush=True,
    )
    return records


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("input_tif", type=Path)
    parser.add_argument("angle_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_batch(
        args.input_tif,
        args.angle_csv,
        args.output_dir,
        workers=args.workers,
        overwrite=args.overwrite,
    )

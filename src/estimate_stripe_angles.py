"""Estimate per-slice stripe angles and their smoothed correction angles."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, rotate
import tifffile as tiff

try:
    from preprocess import preprocess_image
except ImportError:
    from .preprocess import preprocess_image


CANDIDATE_ANGLES = np.arange(-4.0, 4.0001, 0.025)


def _set_below_normal_priority():
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000
            )
        except Exception:
            pass


def estimate_stripe_angle(image):
    """Return the correction angle maximizing the horizontal stripe profile."""
    roi = np.asarray(image[1400:3000:4, 400:1500:4], dtype=np.float32)
    highpass = roi - gaussian_filter(roi, sigma=(7.5, 7.5))
    highpass *= np.outer(np.hanning(highpass.shape[0]), np.hanning(highpass.shape[1]))
    scores = []
    for angle in CANDIDATE_ANGLES:
        aligned = rotate(
            highpass,
            angle=angle,
            reshape=False,
            order=1,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        profile = np.median(aligned[30:-30, 30:-30], axis=1)
        scores.append(np.std(profile))
    scores = np.asarray(scores)
    best_index = int(np.argmax(scores))
    confidence = float(scores[best_index] / np.median(scores))
    return (
        float(CANDIDATE_ANGLES[best_index]),
        float(scores[best_index]),
        confidence,
    )


def _orient_for_horizontal_stripes(raw):
    """Transpose landscape acquisitions so acquisition streaks run by row."""
    raw = np.asarray(raw)
    if raw.shape[1] > raw.shape[0]:
        return raw.T
    return raw


def _prepare_estimation_image(raw, preprocess_first):
    raw = _orient_for_horizontal_stripes(raw)
    if not preprocess_first:
        return raw
    result = preprocess_image(
        raw,
        shading_sigma=120,
        shading_reference_level=100,
        shading_max_gain=3.0,
        stripe_sigma=(128, 256),
        stripe_level=7,
        stripe_wavelet="db2",
        bg_radius=20,
        return_intermediates=False,
    )
    return np.clip(result["preprocessed"] * 100.0, 0, 65535).astype(np.uint16)


def _estimate_chunk(args):
    input_tif, page_indices, preprocess_first = args
    _set_below_normal_priority()
    rows = []
    with tiff.TiffFile(str(input_tif)) as tif:
        for page_index in page_indices:
            raw = tif.pages[page_index].asarray()
            image = _prepare_estimation_image(raw, preprocess_first)
            angle, score, confidence = estimate_stripe_angle(image)
            rows.append(
                {
                    "slice": page_index + 1,
                    "file": "{}_{:03d}.tif".format(input_tif.stem, page_index + 1),
                    "angle_deg": angle,
                    "alignment_score": score,
                    "confidence_ratio": confidence,
                }
            )
    return rows


def estimate_stack(input_tif, output_csv, workers=12, preprocess_first=False):
    input_tif = Path(input_tif)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with tiff.TiffFile(str(input_tif)) as tif:
        page_count = len(tif.pages)
    workers = max(1, min(int(workers), page_count))
    page_indices = list(range(page_count))
    chunk_size = (page_count + workers - 1) // workers
    chunks = [
        page_indices[index : index + chunk_size]
        for index in range(0, page_count, chunk_size)
    ]

    print(
        "Estimating {} slices with {} below-normal-priority workers.".format(
            page_count, workers
        ),
        flush=True,
    )
    started = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _estimate_chunk, (input_tif, chunk, bool(preprocess_first))
            )
            for chunk in chunks
        ]
        for future in as_completed(futures):
            rows.extend(future.result())
            print(
                "Estimated {}/{} slices ({:.1f} min)".format(
                    len(rows), page_count, (time.time() - started) / 60.0
                ),
                flush=True,
            )

    angles = pd.DataFrame(rows).sort_values("slice").reset_index(drop=True)
    angles["recommended_angle_deg"] = angles["angle_deg"].rolling(
        window=11, center=True, min_periods=1
    ).median()
    angles["global_median_deg"] = float(angles["angle_deg"].median())
    angles.to_csv(str(output_csv), index=False)
    print("Saved: {}".format(output_csv), flush=True)
    print(angles["angle_deg"].describe().to_string(), flush=True)
    return angles


def main():
    parser = ArgumentParser()
    parser.add_argument("input_tif", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--preprocess-first", action="store_true")
    args = parser.parse_args()
    estimate_stack(
        args.input_tif,
        args.output_csv,
        workers=args.workers,
        preprocess_first=args.preprocess_first,
    )


if __name__ == "__main__":
    main()

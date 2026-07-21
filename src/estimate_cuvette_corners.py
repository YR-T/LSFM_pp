"""Estimate cuvette inner-wall corners independently for every image slice."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
import re
from threading import Lock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian
from skimage.measure import block_reduce
from skimage.transform import hough_line, hough_line_peaks
import tifffile as tiff


DATASET_PATTERN = re.compile(r"^MAY(?P<day>\d+)(?P<hemisphere>[LR])_FOS_(?P<view>[dv])$")
PLOT_LOCK = Lock()


def current_path(project_root, path_string):
    path = Path(path_string)
    parts = list(path.parts)
    if "outputs" in parts:
        return project_root.joinpath(*parts[parts.index("outputs"):])
    return path


def expected_corner_sides(dataset):
    match = DATASET_PATTERN.match(dataset)
    if match is None:
        raise ValueError("Unexpected dataset name: {}".format(dataset))
    day = int(match.group("day"))
    hemisphere = match.group("hemisphere")
    view = match.group("view")
    if day == 8:
        mapping = {
            ("L", "d"): ("top", "right"),
            ("L", "v"): ("top", "left"),
            ("R", "d"): ("top", "right"),
            ("R", "v"): ("bottom", "right"),
        }
        return mapping[(hemisphere, view)]
    long_side = (
        "bottom" if (hemisphere, view) in {("L", "d"), ("R", "v")} else "top"
    )
    return long_side, "left"


def estimate_one(image, downsample=8, outer_fraction=0.20, angle_limit=20.0):
    reduced = block_reduce(
        image,
        block_size=(downsample, downsample),
        func=np.mean,
    ).astype(np.float32)
    transformed = gaussian(np.log1p(reduced), sigma=1.5, preserve_range=True)
    gradient_y = ndi.sobel(transformed, axis=0)
    gradient_x = ndi.sobel(transformed, axis=1)
    edge_strength = np.hypot(gradient_x, gradient_y)
    height, width = transformed.shape
    return transformed, gradient_x, gradient_y, edge_strength, height, width


def estimate_corner(
    image,
    long_side,
    short_side,
    downsample=8,
    outer_fraction=0.20,
    angle_limit=20.0,
):
    (
        transformed,
        gradient_x,
        gradient_y,
        edge_strength,
        height,
        width,
    ) = estimate_one(image, downsample, outer_fraction, angle_limit)

    threshold = float(np.percentile(edge_strength, 92.0))
    edges = edge_strength >= threshold
    band = max(4, int(round(height * outer_fraction)))
    wall_band = np.zeros_like(edges)
    if long_side == "top":
        wall_band[:band, :] = True
    else:
        wall_band[height - band:, :] = True
    edges &= wall_band

    # A horizontal-ish line has a normal close to +/- 90 degrees.
    theta_degrees = np.concatenate(
        [
            np.linspace(-90.0, -90.0 + angle_limit, 81),
            np.linspace(90.0 - angle_limit, 89.75, 80),
        ]
    )
    hspace, theta, distance = hough_line(edges, theta=np.deg2rad(theta_degrees))
    peak_values, peak_angles, peak_distances = hough_line_peaks(
        hspace,
        theta,
        distance,
        threshold=max(8, int(round(width * 0.06))),
        min_distance=6,
        min_angle=4,
        num_peaks=30,
    )

    candidates = []
    center_x = (width - 1) / 2.0
    for votes, normal_angle, line_distance in zip(
        peak_values, peak_angles, peak_distances
    ):
        sin_theta = math.sin(float(normal_angle))
        if abs(sin_theta) < 1e-6:
            continue
        center_y = (
            float(line_distance) - center_x * math.cos(float(normal_angle))
        ) / sin_theta
        if long_side == "top" and not (0 <= center_y < band):
            continue
        if long_side == "bottom" and not (height - band <= center_y < height):
            continue

        # Direction vector along the detected wall; force it toward +X.
        ux = -math.sin(float(normal_angle))
        uy = math.cos(float(normal_angle))
        if ux < 0:
            ux, uy = -ux, -uy
        angle_degrees = math.degrees(math.atan2(uy, ux))
        border_distance = center_y if long_side == "top" else height - 1 - center_y
        orientation_score = float(votes) / (
            1.0 + 0.01 * max(0.0, border_distance)
        )
        candidates.append(
            (
                orientation_score,
                float(votes),
                angle_degrees,
                ux,
                uy,
                center_x,
                center_y,
            )
        )

    if not candidates:
        return {
            "success": False,
            "failure": "no_long_wall_line",
        }

    # Estimate orientation from the clear border-spanning wall first. Position
    # is selected separately so parallel inner/outer surfaces do not change
    # the otherwise reliable angle estimate.
    orientation = max(candidates, key=lambda candidate: candidate[0])
    _, long_votes, angle_degrees, ux, uy, point_x, point_y = orientation
    if long_side == "top":
        maximum_long_votes = max(candidate[1] for candidate in candidates)
        parallel_inner_candidates = [
            candidate for candidate in candidates
            if candidate[1] >= 0.35 * maximum_long_votes
            and abs(candidate[2] - angle_degrees) <= 0.75
        ]
        if parallel_inner_candidates:
            # In this acquisition the upper inner wall is typically near 12%
            # of image height. This distinguishes it from the outer surface
            # near the frame and from deeper specimen contours.
            expected_inner_y = height * 0.12
            point_y = min(
                parallel_inner_candidates,
                key=lambda candidate: abs(candidate[-1] - expected_inner_y),
            )[-1]
    # v is the normal pointing downward for a near-horizontal wall.
    vx, vy = -uy, ux
    q0 = point_x * vx + point_y * vy

    yy, xx = np.indices((height, width), dtype=np.float32)
    s_coordinate = xx * ux + yy * uy
    q_coordinate = xx * vx + yy * vy
    if long_side == "top":
        # With orientation fixed, recover a weak inner-wall offset from the
        # directional-gradient profile. The 12%-height prior is deliberately
        # soft; image evidence still determines the selected bin.
        wall_gradient = np.abs(gradient_x * vx + gradient_y * vy)
        q_min = float(np.min(q_coordinate))
        q_index = np.floor(q_coordinate - q_min).astype(np.int32)
        q_bins = int(math.ceil(float(np.max(q_coordinate)) - q_min)) + 2
        central = (xx >= width * 0.05) & (xx <= width * 0.95)
        q_sums = np.bincount(
            q_index[central], weights=wall_gradient[central], minlength=q_bins
        )
        q_counts = np.bincount(q_index[central], minlength=q_bins)
        q_profile = np.divide(
            q_sums,
            q_counts,
            out=np.zeros_like(q_sums, dtype=np.float64),
            where=q_counts > 0,
        )
        q_profile = ndi.gaussian_filter1d(q_profile, 1.2)
        # Prior applies to the corner itself, not to the wall position at the
        # image center. This distinction matters once the wall is tilted.
        expected_corner_x = width * (0.035 if short_side == "left" else 0.965)
        expected_q = expected_corner_x * vx + height * 0.12 * vy
        expected_bin = expected_q - q_min
        q_low = max(0, int(math.floor(expected_bin - height * 0.06)))
        q_high = min(q_bins, int(math.ceil(expected_bin + height * 0.06)))
        q_candidates = np.arange(q_low, q_high)
        if len(q_candidates):
            q_scores = q_profile[q_candidates] / (
                1.0 + 0.50 * np.abs(q_candidates - expected_bin)
            )
            q0 = q_min + float(q_candidates[int(np.argmax(q_scores))]) + 0.5
    if long_side == "top":
        interior = (q_coordinate >= q0) & (
            q_coordinate <= q0 + height * 0.45
        )
    else:
        interior = (q_coordinate <= q0) & (
            q_coordinate >= q0 - height * 0.45
        )

    # Detect the short wall on the side supplied by the acquisition prior; its
    # final angle is fixed to exactly 90 degrees from the long wall.
    s_min = float(np.min(s_coordinate))
    s_max = float(np.max(s_coordinate))
    if short_side == "left":
        search_start = s_min
        search_end = s_min + 0.20 * (s_max - s_min)
    else:
        search_start = s_min + 0.80 * (s_max - s_min)
        search_end = s_max
    short_region = (
        interior
        & (s_coordinate >= search_start)
        & (s_coordinate <= search_end)
    )
    short_threshold = float(np.percentile(edge_strength[short_region], 85.0))
    short_edges = short_region & (edge_strength >= short_threshold)
    short_theta = np.deg2rad(
        angle_degrees + np.linspace(-4.0, 4.0, 33)
    )
    short_hspace, short_angles, short_distances = hough_line(
        short_edges, theta=short_theta
    )
    short_values, short_peak_angles, short_peak_distances = hough_line_peaks(
        short_hspace,
        short_angles,
        short_distances,
        threshold=max(8, int(round(height * 0.06))),
        min_distance=5,
        min_angle=3,
        num_peaks=20,
    )
    short_candidates = []
    for votes, normal_angle, line_distance in zip(
        short_values, short_peak_angles, short_peak_distances
    ):
        nx, ny = math.cos(float(normal_angle)), math.sin(float(normal_angle))
        matrix = np.asarray([[vx, vy], [nx, ny]], dtype=np.float64)
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) < 1e-6:
            continue
        intersection = np.linalg.solve(
            matrix, np.asarray([q0, float(line_distance)], dtype=np.float64)
        )
        candidate_s = float(intersection[0] * ux + intersection[1] * uy)
        if not (search_start <= candidate_s <= search_end):
            continue
        candidate_score = float(votes)
        short_candidates.append(
            (
                candidate_score,
                float(votes),
                candidate_s,
                float(intersection[0]),
            )
        )
    if not short_candidates:
        return {
            "success": False,
            "failure": "no_short_wall_line",
        }
    maximum_short_votes = max(candidate[1] for candidate in short_candidates)
    if short_side == "left":
        credible_short = [
            candidate for candidate in short_candidates
            if candidate[1] >= 0.15 * maximum_short_votes
            and candidate[3] >= width * 0.015
        ]
    else:
        credible_short = [
            candidate for candidate in short_candidates
            if candidate[1] >= 0.15 * maximum_short_votes
            and candidate[3] <= width * 0.985
        ]
    if not credible_short:
        credible_short = short_candidates
    # Select the first reliable inner wall after the relevant frame edge.
    if short_side == "left":
        _, short_votes, s0, _ = min(
            credible_short, key=lambda candidate: candidate[3]
        )
    else:
        _, short_votes, s0, _ = max(
            credible_short, key=lambda candidate: candidate[3]
        )
    # Refine the short-wall offset using a directional-gradient profile. This
    # retains weak straight walls that fall below Hough peak threshold.
    directional_gradient = np.abs(gradient_x * ux + gradient_y * uy)
    s_index = np.floor(s_coordinate - s_min).astype(np.int32)
    s_bins = int(math.ceil(s_max - s_min)) + 2
    s_sums = np.bincount(
        s_index[short_region],
        weights=directional_gradient[short_region],
        minlength=s_bins,
    )
    s_counts = np.bincount(s_index[short_region], minlength=s_bins)
    s_profile = np.divide(
        s_sums,
        s_counts,
        out=np.zeros_like(s_sums, dtype=np.float64),
        where=s_counts > 0,
    )
    s_profile = ndi.gaussian_filter1d(s_profile, 1.2)
    expected_x = width * (0.035 if short_side == "left" else 0.965)
    expected_y = (q0 - expected_x * vx) / max(vy, 1e-6)
    expected_s = expected_x * ux + expected_y * uy
    expected_s_bin = expected_s - s_min
    s_low = max(0, int(math.floor(expected_s_bin - width * 0.035)))
    s_high = min(s_bins, int(math.ceil(expected_s_bin + width * 0.035)))
    s_candidates = np.arange(s_low, s_high)
    if len(s_candidates):
        s_scores = s_profile[s_candidates] / (
            1.0 + 0.10 * np.abs(s_candidates - expected_s_bin)
        )
        s0 = s_min + float(s_candidates[int(np.argmax(s_scores))]) + 0.5
    corner_x = s0 * ux + q0 * vx
    corner_y = s0 * uy + q0 * vy

    return {
        "success": True,
        "failure": "",
        "angle_degrees": angle_degrees,
        "corner_x": corner_x * downsample,
        "corner_y": corner_y * downsample,
        "long_wall_votes": long_votes,
        "short_wall_score": short_votes,
        "short_wall_votes": short_votes,
        "long_q_lowres": q0,
        "short_s_lowres": s0,
        "ux": ux,
        "uy": uy,
        "vx": vx,
        "vy": vy,
        "downsample": downsample,
    }


def summarize(rows):
    valid = [row for row in rows if row["success"]]
    if not valid:
        raise RuntimeError("No valid cuvette-corner estimates")
    result = {"valid_slices": len(valid), "total_slices": len(rows)}
    for field in ["angle_degrees", "corner_x", "corner_y"]:
        values = np.asarray([float(row[field]) for row in valid], dtype=np.float64)
        result[field + "_median"] = float(np.median(values))
        result[field + "_q25"] = float(np.percentile(values, 25))
        result[field + "_q75"] = float(np.percentile(values, 75))
    result["long_wall_votes_median"] = float(
        np.median([float(row["long_wall_votes"]) for row in valid])
    )
    result["short_wall_score_median"] = float(
        np.median([float(row["short_wall_score"]) for row in valid])
    )
    return result


def save_qc(output_dir, dataset, maximum_projection, rows, summary, long_side):
    low, high = np.percentile(maximum_projection, [1.0, 99.7])
    height, width = maximum_projection.shape
    angle = math.radians(summary["angle_degrees_median"])
    ux, uy = math.cos(angle), math.sin(angle)
    vx, vy = -uy, ux
    corner = np.asarray(
        [summary["corner_x_median"], summary["corner_y_median"]], dtype=float
    )
    extent = max(height, width) * 2.0
    long_points = np.vstack([corner - extent * np.array([ux, uy]), corner + extent * np.array([ux, uy])])
    short_points = np.vstack([corner - extent * np.array([vx, vy]), corner + extent * np.array([vx, vy])])
    valid = [row for row in rows if row["success"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=160)
    axes[0].imshow(maximum_projection, cmap="gray", vmin=low, vmax=high)
    axes[0].plot(long_points[:, 0], long_points[:, 1], color="cyan", lw=1.3, label="long wall")
    axes[0].plot(short_points[:, 0], short_points[:, 1], color="yellow", lw=1.3, label="orthogonal short wall")
    axes[0].scatter(corner[0], corner[1], s=30, c="red", marker="x", label="median corner")
    axes[0].set_title("Median walls on maximum projection")
    axes[0].legend(loc="best", fontsize=7)
    axes[0].set_xlim(0, width)
    axes[0].set_ylim(height, 0)
    axes[0].set_axis_off()

    axes[1].scatter(
        [row["corner_x"] for row in valid],
        [row["corner_y"] for row in valid],
        c=[row["slice"] for row in valid],
        s=10,
        cmap="viridis",
        alpha=0.65,
    )
    axes[1].scatter(corner[0], corner[1], s=60, c="red", marker="x")
    axes[1].set_title("Per-slice corners (color = slice)")
    axes[1].set_xlim(0, width)
    axes[1].set_ylim(height, 0)
    axes[1].set_aspect("equal")
    axes[1].grid(alpha=0.2)
    fig.suptitle(
        "{} | {} wall | angle={:.2f} deg | valid={}/{}".format(
            dataset,
            long_side,
            summary["angle_degrees_median"],
            summary["valid_slices"],
            summary["total_slices"],
        )
    )
    fig.tight_layout()
    fig.savefig(output_dir / "cuvette_corner_qc.png", bbox_inches="tight")
    plt.close(fig)


def process_dataset(dataset, image_paths, output_root, downsample):
    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    long_side, short_side = expected_corner_sides(dataset)
    rows = []
    maximum_projection = None
    for index, image_path in enumerate(image_paths, start=1):
        image = tiff.imread(str(image_path))
        if maximum_projection is None:
            maximum_projection = image.copy()
        else:
            np.maximum(maximum_projection, image, out=maximum_projection)
        estimate = estimate_corner(
            image, long_side, short_side, downsample=downsample
        )
        estimate.update(
            {
                "dataset": dataset,
                "slice": index,
                "input_tif": str(image_path),
                "long_wall_side": long_side,
                "short_wall_side": short_side,
            }
        )
        rows.append(estimate)

    fields = sorted({key for row in rows for key in row})
    with (output_dir / "cuvette_corner_by_slice.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary.update(
        {
            "dataset": dataset,
            "long_wall_side": long_side,
            "short_wall_side": short_side,
            "downsample": downsample,
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    # Matplotlib is not thread-safe even with the non-GUI backend.
    with PLOT_LOCK:
        save_qc(output_dir, dataset, maximum_projection, rows, summary, long_side)
    return summary


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
        help="Defaults to <result-root>/cuvette_corner_estimation_MAY17plus_left.",
    )
    parser.add_argument("--minimum-day", type=int, default=17)
    parser.add_argument("--maximum-day", type=int)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output_root = args.output_root or (
        args.result_root / "cuvette_corner_estimation_MAY17plus_left"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    grouped = {}
    with (args.result_root / "robust_z_slice_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            match = DATASET_PATTERN.match(row["dataset"])
            if match is None or int(match.group("day")) < args.minimum_day:
                continue
            if (
                args.maximum_day is not None
                and int(match.group("day")) > args.maximum_day
            ):
                continue
            if args.dataset and row["dataset"] not in args.dataset:
                continue
            grouped.setdefault(row["dataset"], []).append(
                (int(row["slice"]), current_path(project_root, row["input_tif"]))
            )
    for dataset in grouped:
        grouped[dataset] = [path for _, path in sorted(grouped[dataset])]
        if len(grouped[dataset]) != 180:
            raise RuntimeError(
                "Expected 180 images for {}, found {}".format(
                    dataset, len(grouped[dataset])
                )
            )

    summaries = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_dataset,
                dataset,
                image_paths,
                output_root,
                args.downsample,
            ): dataset
            for dataset, image_paths in sorted(grouped.items())
        }
        for index, future in enumerate(as_completed(futures), start=1):
            dataset = futures[future]
            summary = future.result()
            summaries.append(summary)
            print(
                "[{}/{}] {} angle={:.2f} corner=({:.1f}, {:.1f}) valid={}/{}".format(
                    index,
                    len(futures),
                    dataset,
                    summary["angle_degrees_median"],
                    summary["corner_x_median"],
                    summary["corner_y_median"],
                    summary["valid_slices"],
                    summary["total_slices"],
                ),
                flush=True,
            )

    summaries.sort(key=lambda row: row["dataset"])
    if summaries:
        fields = sorted({key for row in summaries for key in row})
        with (output_root / "cuvette_corner_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summaries)


if __name__ == "__main__":
    main()

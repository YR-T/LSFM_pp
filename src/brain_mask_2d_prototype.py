"""Build one Z-invariant 2D brain mask and apply it to compact Spot XML."""

from argparse import ArgumentParser
import csv
import json
import math
from pathlib import Path
import re
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import block_reduce, label
from skimage.morphology import binary_closing, binary_dilation, disk
from skimage.transform import resize
import tifffile as tiff


X_PATTERN = re.compile(r'\bPOSITION_X="([^"]+)"')
Y_PATTERN = re.compile(r'\bPOSITION_Y="([^"]+)"')


def largest_component(mask, minimum_area=0):
    labels = label(mask, connectivity=2)
    counts = np.bincount(labels.ravel())
    if len(counts) <= 1:
        return np.zeros_like(mask, dtype=bool), 0
    counts[0] = 0
    component = int(np.argmax(counts))
    area = int(counts[component])
    if area < int(minimum_area):
        return np.zeros_like(mask, dtype=bool), area
    return labels == component, area


def current_path(project_root, path_string):
    path = Path(path_string)
    parts = list(path.parts)
    if "outputs" in parts:
        return project_root.joinpath(*parts[parts.index("outputs"):])
    return path


def detect_cuvette_roi(projection, downsample, wall_margin_pixels):
    reduced = block_reduce(
        projection,
        block_size=(downsample, downsample),
        func=np.mean,
    ).astype(np.float32)
    transformed = np.log1p(reduced)
    gradient_x = np.abs(ndi.sobel(transformed, axis=1))
    gradient_y = np.abs(ndi.sobel(transformed, axis=0))
    vertical_score = ndi.gaussian_filter1d(gradient_x.mean(axis=0), 2.0)
    horizontal_score = ndi.gaussian_filter1d(gradient_y.mean(axis=1), 2.0)
    height, width = reduced.shape

    def strongest_outer(score, size):
        margin = max(2, int(size * 0.02))
        # A cuvette wall is expected close to the image border.  A wider search
        # can mistake a long brain outline for the wall and crop the specimen.
        outer = max(margin + 1, int(size * 0.15))
        candidates = list(range(margin, outer)) + list(
            range(size - outer, size - margin)
        )
        return max(candidates, key=lambda index: score[index])

    vertical = strongest_outer(vertical_score, width)
    horizontal = strongest_outer(horizontal_score, height)
    margin_lowres = int(math.ceil(float(wall_margin_pixels) / downsample))
    roi = np.ones((height, width), dtype=bool)
    if vertical < width // 2:
        roi[:, : min(width, vertical + margin_lowres)] = False
        vertical_side = "left"
    else:
        roi[:, max(0, vertical - margin_lowres + 1):] = False
        vertical_side = "right"
    if horizontal < height // 2:
        roi[: min(height, horizontal + margin_lowres), :] = False
        horizontal_side = "top"
    else:
        roi[max(0, horizontal - margin_lowres + 1):, :] = False
        horizontal_side = "bottom"
    metadata = {
        "vertical_edge_lowres": int(vertical),
        "horizontal_edge_lowres": int(horizontal),
        "vertical_edge_pixel": int(vertical * downsample),
        "horizontal_edge_pixel": int(horizontal * downsample),
        "vertical_side": vertical_side,
        "horizontal_side": horizontal_side,
        "wall_margin_pixels_effective": margin_lowres * downsample,
        "vertical_score": float(vertical_score[vertical]),
        "horizontal_score": float(horizontal_score[horizontal]),
    }
    return roi, metadata


def build_mask(
    image_paths,
    downsample,
    sigma,
    closing_radius,
    dilation_pixels,
    use_cuvette_roi=False,
    wall_margin_pixels=16,
):
    maximum_projection = None
    slice_rows = []
    for index, image_path in enumerate(image_paths, start=1):
        image = tiff.imread(str(image_path))
        if image.ndim != 2:
            raise RuntimeError("Expected a 2D TIFF: {}".format(image_path))
        if maximum_projection is None:
            maximum_projection = image.copy()
        else:
            np.maximum(maximum_projection, image, out=maximum_projection)
        slice_rows.append(
            {
                "slice": index,
                "intensity_median": float(np.median(image)),
                "intensity_p99": float(np.percentile(image, 99.0)),
            }
        )
        if index % 20 == 0 or index == len(image_paths):
            print("[MASK {}/{}]".format(index, len(image_paths)), flush=True)

    reduced = block_reduce(
        maximum_projection,
        block_size=(downsample, downsample),
        func=np.mean,
    ).astype(np.float32)
    if use_cuvette_roi:
        cuvette_roi, cuvette_metadata = detect_cuvette_roi(
            maximum_projection, downsample, wall_margin_pixels
        )
    else:
        cuvette_roi = np.ones_like(reduced, dtype=bool)
        cuvette_metadata = None
    # Compress the very bright cuvette/specimen edges before thresholding.
    # Without this, Otsu can select only the bright surface while excluding a
    # comparatively dim but valid brain interior.
    smoothed = gaussian(np.log1p(reduced), sigma=sigma, preserve_range=True)
    projection_threshold = float(threshold_otsu(smoothed[cuvette_roi]))
    candidate = (smoothed > projection_threshold) & cuvette_roi
    candidate = binary_closing(candidate, disk(closing_radius))
    candidate, component_area = largest_component(
        candidate, int(candidate.size * 0.01)
    )
    candidate = ndi.binary_fill_holes(candidate)
    dilation_lowres = int(math.ceil(float(dilation_pixels) / downsample))
    candidate = binary_dilation(candidate, disk(dilation_lowres))
    candidate &= cuvette_roi
    full_shape = maximum_projection.shape
    full_mask = resize(
        candidate.astype(np.uint8),
        full_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(bool)
    full_cuvette_roi = resize(
        cuvette_roi.astype(np.uint8),
        full_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(bool)
    return (
        full_mask,
        full_cuvette_roi,
        maximum_projection,
        slice_rows,
        component_area,
        projection_threshold,
        cuvette_metadata,
    )


def apply_mask(source_xml, output_xml, mask, sample_stride, write_xml=True):
    temporary = None
    output = None
    if write_xml:
        output_xml.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_xml.with_suffix(".tmp")
        output = temporary.open("w", encoding="utf-8", newline="\n")
        output.write('<?xml version="1.0" encoding="UTF-8"?>\n<spots>\n')
    before = 0
    kept = 0
    outside_image = 0
    sample_before = []
    sample_after = []
    height, width = mask.shape
    try:
        with source_xml.open("r", encoding="utf-8") as input_stream:
            for line in input_stream:
                if "<Spot " not in line:
                    continue
                x_match = X_PATTERN.search(line)
                y_match = Y_PATTERN.search(line)
                if x_match is None or y_match is None:
                    raise ValueError("Spot line is missing X or Y.")
                x = float(x_match.group(1))
                y = float(y_match.group(1))
                xi = int(round(x))
                yi = int(round(y))
                inside_image = 0 <= xi < width and 0 <= yi < height
                inside_mask = inside_image and bool(mask[yi, xi])
                if before % int(sample_stride) == 0:
                    sample_before.append((x, y))
                    if inside_mask:
                        sample_after.append((x, y))
                if inside_mask:
                    if output is not None:
                        output.write(line)
                    kept += 1
                elif not inside_image:
                    outside_image += 1
                before += 1
        if output is not None:
            output.write("</spots>\n")
            output.close()
            output = None
            temporary.replace(output_xml)
    finally:
        if output is not None:
            output.close()
    return {
        "spots_before": before,
        "spots_kept": kept,
        "spots_removed": before - kept,
        "removed_fraction": (before - kept) / float(before) if before else 0.0,
        "spots_outside_image": outside_image,
        "sample_before": np.asarray(sample_before, dtype=np.float32),
        "sample_after": np.asarray(sample_after, dtype=np.float32),
    }


def save_qc(output_dir, projection, mask, result, dataset, cuvette_roi=None):
    low, high = np.percentile(projection, [1.0, 99.7])
    boundary = mask ^ ndi.binary_erosion(mask)
    before = result["sample_before"]
    after = result["sample_after"]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=140)
    axes[0].imshow(projection, cmap="gray", vmin=low, vmax=high)
    if cuvette_roi is not None and not np.all(cuvette_roi):
        axes[0].contour(cuvette_roi, levels=[0.5], colors="cyan", linewidths=0.8)
    axes[0].contour(mask, levels=[0.5], colors="lime", linewidths=0.8)
    axes[0].set_title("Projection + cuvette ROI (cyan) + brain mask (green)")
    axes[1].imshow(projection, cmap="gray", vmin=low, vmax=high)
    if len(before):
        axes[1].scatter(before[:, 0], before[:, 1], s=0.3, c="red", alpha=0.3)
    axes[1].set_title("Before: {:,} spots (sampled)".format(result["spots_before"]))
    axes[2].imshow(projection, cmap="gray", vmin=low, vmax=high)
    if len(after):
        axes[2].scatter(after[:, 0], after[:, 1], s=0.3, c="lime", alpha=0.3)
    axes[2].set_title(
        "After: {:,} ({:.1%} kept)".format(
            result["spots_kept"],
            result["spots_kept"] / float(result["spots_before"]),
        )
    )
    for axis in axes:
        axis.set_xlim(0, projection.shape[1])
        axis.set_ylim(projection.shape[0], 0)
        axis.set_axis_off()
    fig.suptitle(dataset)
    fig.tight_layout()
    fig.savefig(str(output_dir / "mask_and_spots_qc.png"), bbox_inches="tight")
    plt.close(fig)

    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[boundary, 1] = 255
    overlay[boundary, 3] = 255
    tiff.imwrite(str(output_dir / "brain_mask_2d_uint8.tif"), mask.astype(np.uint8) * 255)
    if cuvette_roi is not None:
        tiff.imwrite(
            str(output_dir / "cuvette_roi_2d_uint8.tif"),
            cuvette_roi.astype(np.uint8) * 255,
        )


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="MAY08R_FOS_v")
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--closing-radius", type=int, default=5)
    parser.add_argument("--dilation-pixels", type=int, default=64)
    parser.add_argument("--cuvette-crop", action="store_true")
    parser.add_argument("--wall-margin-pixels", type=int, default=16)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument(
        "--mask-only",
        action="store_true",
        help="Create the mask and QC without writing a masked Spot XML.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=project_root / "outputs" / "TrackMate_final_r2p5_q150",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to <result-root>/mask_2d_prototype.",
    )
    args = parser.parse_args()
    summary_path = args.result_root / "robust_z_slice_summary.csv"
    with summary_path.open("r", encoding="utf-8", newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row["dataset"] == args.dataset
        ]
    rows.sort(key=lambda row: int(row["slice"]))
    if len(rows) != 180:
        raise RuntimeError("Expected 180 slices for {}, found {}".format(args.dataset, len(rows)))
    image_paths = [current_path(project_root, row["input_tif"]) for row in rows]
    shared_manifest = args.result_root / "shared_IDXYZ_xy3" / "manifest.csv"
    with shared_manifest.open("r", encoding="utf-8", newline="") as stream:
        shared_rows = [
            row for row in csv.DictReader(stream)
            if row["dataset"] == args.dataset
        ]
    if len(shared_rows) != 1:
        raise RuntimeError("Shared XML was not found for {}".format(args.dataset))
    source_xml = current_path(project_root, shared_rows[0]["output_xml"])
    output_root = args.output_root or args.result_root / "mask_2d_prototype"
    output_dir = output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_xml = output_dir / (source_xml.stem + "_mask2d.xml")

    started = time.perf_counter()
    (
        mask,
        cuvette_roi,
        projection,
        slice_rows,
        component_area,
        projection_threshold,
        cuvette_metadata,
    ) = build_mask(
        image_paths,
        args.downsample,
        args.sigma,
        args.closing_radius,
        args.dilation_pixels,
        use_cuvette_roi=args.cuvette_crop,
        wall_margin_pixels=args.wall_margin_pixels,
    )
    result = apply_mask(
        source_xml,
        output_xml,
        mask,
        args.sample_stride,
        write_xml=not args.mask_only,
    )
    save_qc(output_dir, projection, mask, result, args.dataset, cuvette_roi)
    with (output_dir / "slice_mask_statistics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(slice_rows[0]))
        writer.writeheader()
        writer.writerows(slice_rows)
    payload = {
        "dataset": args.dataset,
        "method": "maximum_projection_log1p_blur_otsu_largest_component",
        "intensity_transform": "log1p",
        "cuvette_crop": args.cuvette_crop,
        "cuvette": cuvette_metadata,
        "downsample": args.downsample,
        "gaussian_sigma_lowres": args.sigma,
        "closing_radius_lowres": args.closing_radius,
        "dilation_pixels_requested": args.dilation_pixels,
        "dilation_pixels_effective": int(math.ceil(args.dilation_pixels / args.downsample) * args.downsample),
        "mask_pixels": int(mask.sum()),
        "mask_fraction": float(mask.mean()),
        "projection_otsu_threshold": projection_threshold,
        "component_area_lowres_before_fill_and_dilation": int(component_area),
        "spots_before": result["spots_before"],
        "spots_kept": result["spots_kept"],
        "spots_removed": result["spots_removed"],
        "removed_fraction": result["removed_fraction"],
        "source_xml": str(source_xml),
        "output_xml": str(output_xml) if not args.mask_only else None,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

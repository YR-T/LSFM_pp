"""Build a canonical brain-mask prior from corner-aligned datasets."""

from argparse import ArgumentParser
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import block_reduce, label
from skimage.morphology import binary_closing, binary_dilation, disk
from skimage.transform import resize
import tifffile as tiff


def current_path(project_root, path_string):
    path = Path(path_string)
    parts = list(path.parts)
    if "outputs" in parts:
        return project_root.joinpath(*parts[parts.index("outputs"):])
    return path


def load_corner_summaries(paths):
    result = {}
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                result[row["dataset"]] = row
    return result


def component_from_seed(candidate, seed):
    labels = label(candidate, connectivity=2)
    count = int(labels.max())
    if count == 0:
        return np.zeros_like(candidate, dtype=bool), 0, 0
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    overlaps = np.bincount(labels[seed].ravel(), minlength=count + 1)
    areas[0] = 0
    overlaps[0] = 0
    components = np.arange(1, count + 1)
    # Seed overlap is primary; component area resolves close candidates.
    selected = int(
        max(components, key=lambda item: (overlaps[item], areas[item]))
    )
    return labels == selected, int(areas[selected]), int(overlaps[selected])


def erode_toward_upper_left(mask, radius):
    """Shrink bottom/right boundaries while preserving top/left anchors."""
    result = mask.copy()
    height, width = mask.shape
    for dy in range(radius + 1):
        maximum_dx = int(math.floor(math.sqrt(max(0, radius * radius - dy * dy))))
        for dx in range(maximum_dx + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.zeros_like(mask)
            shifted[: height - dy, : width - dx] = mask[dy:, dx:]
            result &= shifted
    return result


def canonical_basis(corner_row):
    angle = math.radians(float(corner_row["angle_degrees_median"]))
    wall_u = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    downward_normal = np.asarray([-math.sin(angle), math.cos(angle)], dtype=np.float64)
    if corner_row["short_wall_side"] == "left":
        inward_x = wall_u
    else:
        inward_x = -wall_u
    if corner_row["long_wall_side"] == "top":
        inward_y = downward_normal
    else:
        inward_y = -downward_normal
    return inward_x, inward_y


def align_projection(projection, corner_row, output_shape, downsample):
    reduced = block_reduce(
        projection,
        block_size=(downsample, downsample),
        func=np.mean,
    ).astype(np.float32)
    corner = np.asarray(
        [
            float(corner_row["corner_x_median"]) / downsample,
            float(corner_row["corner_y_median"]) / downsample,
        ],
        dtype=np.float64,
    )
    inward_x, inward_y = canonical_basis(corner_row)
    canonical_y, canonical_x = np.indices(output_shape, dtype=np.float32)
    source_x = corner[0] + canonical_x * inward_x[0] + canonical_y * inward_y[0]
    source_y = corner[1] + canonical_x * inward_x[1] + canonical_y * inward_y[1]
    valid = (
        (source_x >= 0)
        & (source_x <= reduced.shape[1] - 1)
        & (source_y >= 0)
        & (source_y <= reduced.shape[0] - 1)
    )
    aligned = ndi.map_coordinates(
        reduced,
        [source_y, source_x],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32)
    transformed = np.log1p(aligned)
    values = transformed[valid]
    low, high = np.percentile(values, [5.0, 99.5])
    normalized = np.clip((transformed - low) / max(high - low, 1e-6), 0, 1)
    normalized[~valid] = 0
    return normalized.astype(np.float32), valid


def individual_mask(
    normalized,
    valid,
    seed,
    gaussian_sigma,
    closing_radius,
    dilation_radius,
    wall_exclusion_radius,
):
    smoothed = gaussian(
        normalized, sigma=gaussian_sigma, preserve_range=True
    )
    threshold = float(threshold_otsu(smoothed[valid]))
    candidate = (smoothed > threshold) & valid
    # Remove the two canonical wall bands before component selection. The
    # final dilation restores genuine brain-to-wall contact locally without
    # retaining the complete bright cuvette walls.
    candidate[:wall_exclusion_radius, :] = False
    candidate[:, :wall_exclusion_radius] = False
    # Rv heads can extend into the upper-right, but the lower part of the far
    # right edge is an opposite-wall artifact in several acquisitions.
    candidate[int(candidate.shape[0] * 0.45):, int(candidate.shape[1] * 0.90):] = False
    candidate = binary_closing(candidate, disk(closing_radius))
    component, area, overlap = component_from_seed(candidate, seed & valid)
    component = ndi.binary_fill_holes(component)
    component = binary_dilation(component, disk(dilation_radius)) & valid
    return component, threshold, area, overlap


def process_dataset(
    dataset,
    image_paths,
    corner_row,
    output_shape,
    downsample,
    seed,
    wall_exclusion_pixels,
):
    maximum_projection = None
    for image_path in image_paths:
        image = tiff.imread(str(image_path))
        if maximum_projection is None:
            maximum_projection = image.copy()
        else:
            np.maximum(maximum_projection, image, out=maximum_projection)
    normalized, valid = align_projection(
        maximum_projection, corner_row, output_shape, downsample
    )
    mask, threshold, area, overlap = individual_mask(
        normalized,
        valid,
        seed,
        gaussian_sigma=4.0,
        closing_radius=5,
        dilation_radius=int(math.ceil(64.0 / downsample)),
        wall_exclusion_radius=int(math.ceil(wall_exclusion_pixels / downsample)),
    )
    return {
        "dataset": dataset,
        "normalized": normalized,
        "valid": valid,
        "mask": mask,
        "otsu_threshold": threshold,
        "component_area_lowres": area,
        "seed_overlap_lowres": overlap,
        "mask_fraction_valid": float(mask.sum() / max(valid.sum(), 1)),
    }


def save_qc(
    output_dir,
    results,
    probability,
    prior_mask,
    seed,
    downsample,
    base_mask=None,
    coverage_mask=None,
    prior_title="Final Rv prior",
    placement="Rv",
):
    prefix = placement.lower()
    columns = 3
    rows = int(math.ceil(len(results) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4.5 * rows), dpi=150)
    axes = np.asarray(axes).reshape(-1)
    for axis, result in zip(axes, results):
        axis.imshow(result["normalized"], cmap="gray", vmin=0, vmax=1)
        axis.contour(result["mask"], levels=[0.5], colors="lime", linewidths=0.7)
        axis.contour(seed, levels=[0.5], colors="cyan", linewidths=0.5)
        axis.set_title(
            "{} | mask={:.1%}".format(
                result["dataset"], result["mask_fraction_valid"]
            )
        )
        axis.set_axis_off()
    for axis in axes[len(results):]:
        axis.set_axis_off()
    fig.suptitle(
        "Corner-aligned {} projections and individual masks".format(placement)
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / (prefix + "_aligned_individual_masks_qc.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(rows, columns, figsize=(15, 4.5 * rows), dpi=150)
    axes = np.asarray(axes).reshape(-1)
    prior_overlay = np.ma.masked_where(~prior_mask, prior_mask)
    for axis, result in zip(axes, results):
        axis.imshow(result["normalized"], cmap="gray", vmin=0, vmax=1)
        axis.imshow(prior_overlay, cmap="Greens", vmin=0, vmax=1, alpha=0.20)
        axis.contour(prior_mask, levels=[0.5], colors="lime", linewidths=0.9)
        axis.set_title(result["dataset"])
        axis.set_axis_off()
    for axis in axes[len(results):]:
        axis.set_axis_off()
    fig.suptitle(
        "The same {} prior overlaid on all 9 corner-aligned datasets".format(
            placement
        )
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / (prefix + "_prior_overlaid_on_9_qc.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    median_projection = np.median(
        np.stack([result["normalized"] for result in results]), axis=0
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=160)
    axes[0].imshow(median_projection, cmap="gray", vmin=0, vmax=1)
    axes[0].contour(seed, levels=[0.5], colors="cyan", linewidths=0.7)
    axes[0].set_title("Median aligned projection + radius 500 px seed")
    image = axes[1].imshow(probability, cmap="magma", vmin=0, vmax=1)
    axes[1].contour(prior_mask, levels=[0.5], colors="cyan", linewidths=0.8)
    axes[1].set_title("Mask occupancy probability")
    fig.colorbar(image, ax=axes[1], fraction=0.046)
    axes[2].imshow(median_projection, cmap="gray", vmin=0, vmax=1)
    axes[2].contour(prior_mask, levels=[0.5], colors="lime", linewidths=0.9)
    axes[2].set_title(prior_title)
    for axis in axes:
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(
        output_dir / (prefix + "_brain_prior_qc.png"), bbox_inches="tight"
    )
    plt.close(fig)

    if base_mask is not None and coverage_mask is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=160)
        for axis in axes:
            axis.imshow(median_projection, cmap="gray", vmin=0, vmax=1)
            axis.set_axis_off()
        axes[0].contour(base_mask, levels=[0.5], colors="cyan", linewidths=0.9)
        axes[0].set_title("Base: at least 2/9 datasets")
        axes[1].contour(coverage_mask, levels=[0.5], colors="yellow", linewidths=0.9)
        axes[1].set_title("Coverage: at least 1/9 datasets")
        axes[2].contour(prior_mask, levels=[0.5], colors="lime", linewidths=0.9)
        axes[2].set_title("Fixed common prior used for individual adaptation")
        fig.tight_layout()
        fig.savefig(
            output_dir / (prefix + "_prior_envelope_steps_qc.png"),
            bbox_inches="tight",
        )
        plt.close(fig)
    return median_projection


def build_individual_adaptive_masks(
    results,
    base_mask,
    seed,
    downsample,
    support_distance_pixels=256.0,
):
    support_radius = int(math.ceil(support_distance_pixels / downsample))
    near_prior = binary_dilation(base_mask, disk(support_radius))
    adapted = []
    for result in results:
        protected = result["mask"] & near_prior
        # Do not protect the known opposite-wall region merely because it is
        # close to the conservative common prior.
        protected[
            int(protected.shape[0] * 0.45):,
            int(protected.shape[1] * 0.90):,
        ] = False
        # Treat the shared P>=0.2 prior as anatomical support. Low-frequency
        # signal can be locally weak even where brain tissue is present, so an
        # individual image may expand this prior but must never erode it.
        final_mask = base_mask | protected
        final_mask, _, _ = component_from_seed(final_mask, seed)
        final_mask = binary_closing(ndi.binary_fill_holes(final_mask), disk(3))
        adapted.append(
            {
                "dataset": result["dataset"],
                "protected": protected,
                "mask": final_mask,
                "base_fraction": float(base_mask.mean()),
                "final_fraction": float(final_mask.mean()),
                "added_fraction": float((final_mask & ~base_mask).mean()),
                "removed_fraction": float((base_mask & ~final_mask).mean()),
            }
        )
    return adapted


def save_individual_adaptation_qc(
    output_dir, results, adapted, base_mask, title, placement="Rv"
):
    fig, axes = plt.subplots(3, 3, figsize=(15, 13.5), dpi=150)
    axes = axes.reshape(-1)
    for axis, result, item in zip(axes, results, adapted):
        axis.imshow(result["normalized"], cmap="gray", vmin=0, vmax=1)
        axis.contour(base_mask, levels=[0.5], colors="cyan", linewidths=0.55)
        axis.contour(item["mask"], levels=[0.5], colors="lime", linewidths=1.0)
        axis.set_title(
            "{} | +{:.1%} / -{:.1%}".format(
                item["dataset"],
                item["added_fraction"],
                item["removed_fraction"],
            )
        )
        axis.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(
        output_dir
        / (
            placement.lower()
            + "_individual_adapted_masks_overlaid_on_9_qc.png"
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


def build_shared_adaptive_masks(results, shared_mask, reference_mask):
    """Use one common mask for every dataset and report its change."""
    adapted = []
    for result in results:
        adapted.append(
            {
                "dataset": result["dataset"],
                "protected": np.zeros_like(shared_mask, dtype=bool),
                "mask": shared_mask,
                "base_fraction": float(reference_mask.mean()),
                "final_fraction": float(shared_mask.mean()),
                "added_fraction": float((shared_mask & ~reference_mask).mean()),
                "removed_fraction": float((reference_mask & ~shared_mask).mean()),
            }
        )
    return adapted


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
        help="Defaults to <result-root>/brain_mask_prior_Rv_prototype.",
    )
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--seed-radius", type=float, default=500.0)
    parser.add_argument("--probability-threshold", type=float, default=0.20)
    parser.add_argument("--wall-exclusion", type=float, default=40.0)
    parser.add_argument("--directional-erosion", type=float, default=64.0)
    parser.add_argument(
        "--placement", choices=("Ld", "Lv", "Rd", "Rv"), default="Rv"
    )
    parser.add_argument(
        "--prior-mode",
        choices=("expansion-only", "coverage-union", "coverage-eroded"),
        default="expansion-only",
        help=(
            "expansion-only preserves P>=threshold and adds individual foreground; "
            "coverage-union uses the unmodified union of all individual masks; "
            "coverage-eroded unions all individual masks and erodes that shared envelope."
        ),
    )
    args = parser.parse_args()
    output_dir = args.output_root or (
        args.result_root / ("brain_mask_prior_{}_prototype".format(args.placement))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    corner_rows = load_corner_summaries(
        [
            args.result_root
            / "cuvette_corner_estimation_MAY05to08_calibrated"
            / "cuvette_corner_summary.csv",
            args.result_root
            / "cuvette_corner_estimation_MAY17plus_calibrated"
            / "cuvette_corner_summary.csv",
        ]
    )
    placement_suffix = {
        "Ld": "L_FOS_d",
        "Lv": "L_FOS_v",
        "Rd": "R_FOS_d",
        "Rv": "R_FOS_v",
    }[args.placement]
    grouped = {}
    image_shapes = {}
    with (args.result_root / "robust_z_slice_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            dataset = row["dataset"]
            if (
                not dataset.endswith(placement_suffix)
                or dataset.startswith("MAY08")
            ):
                continue
            if dataset not in corner_rows:
                continue
            grouped.setdefault(dataset, []).append(
                (int(row["slice"]), current_path(project_root, row["input_tif"]))
            )
    for dataset in grouped:
        grouped[dataset] = [path for _, path in sorted(grouped[dataset])]
        if len(grouped[dataset]) != 180:
            raise RuntimeError("Expected 180 images for {}".format(dataset))
        with tiff.TiffFile(str(grouped[dataset][0])) as tif:
            image_shapes[dataset] = tuple(tif.pages[0].shape)
    expected = {
        "{}{}".format(day, placement_suffix)
        for day in (
            "MAY05", "MAY06", "MAY07", "MAY17", "MAY18",
            "MAY19", "MAY20", "MAY21", "MAY25",
        )
    }
    if set(grouped) != expected:
        raise RuntimeError(
            "Unexpected {} datasets: {}".format(args.placement, sorted(grouped))
        )
    full_height = max(shape[0] for shape in image_shapes.values())
    full_width = max(shape[1] for shape in image_shapes.values())
    output_shape = (
        int(math.ceil(full_height / args.downsample)),
        int(math.ceil(full_width / args.downsample)),
    )
    yy, xx = np.indices(output_shape, dtype=np.float32)
    seed = np.hypot(xx, yy) <= args.seed_radius / args.downsample

    results = []
    for index, dataset in enumerate(sorted(grouped), start=1):
        result = process_dataset(
            dataset,
            grouped[dataset],
            corner_rows[dataset],
            output_shape,
            args.downsample,
            seed,
            args.wall_exclusion,
        )
        results.append(result)
        print(
            "[{}/{}] {} mask={:.1%} seed_overlap={:,}".format(
                index,
                len(grouped),
                dataset,
                result["mask_fraction_valid"],
                result["seed_overlap_lowres"],
            ),
            flush=True,
        )

    probability = np.mean(
        np.stack([result["mask"].astype(np.float32) for result in results]),
        axis=0,
    )
    base_candidate = probability >= args.probability_threshold
    base_mask, _, _ = component_from_seed(base_candidate, seed)
    base_mask = binary_closing(ndi.binary_fill_holes(base_mask), disk(5))
    coverage_candidate = probability > 0
    # Exclude the canonical opposite-wall zone from the all-example union.
    # Dilation of an individual mask can otherwise reconnect a thin cuvette
    # wall artifact and turn it into a large shared-prior appendage.
    coverage_candidate[
        int(coverage_candidate.shape[0] * 0.45):,
        int(coverage_candidate.shape[1] * 0.90):,
    ] = False
    coverage_mask, _, _ = component_from_seed(coverage_candidate, seed)
    coverage_mask = binary_closing(ndi.binary_fill_holes(coverage_mask), disk(5))
    erosion_lowres = int(math.ceil(args.directional_erosion / args.downsample))
    directional_coverage = erode_toward_upper_left(
        coverage_mask, erosion_lowres
    )
    if args.prior_mode in {"coverage-union", "coverage-eroded"}:
        # Make one common envelope that includes every observed individual,
        # optionally pulling only its bottom/right boundary upper-left.
        prior_mask = (
            coverage_mask
            if args.prior_mode == "coverage-union"
            else directional_coverage
        )
        adaptation_reference = coverage_mask
    else:
        prior_mask = base_mask
        adaptation_reference = base_mask
    prior_mask, _, _ = component_from_seed(prior_mask, seed)
    prior_mask = binary_closing(ndi.binary_fill_holes(prior_mask), disk(5))
    median_projection = save_qc(
        output_dir,
        results,
        probability,
        prior_mask,
        seed,
        args.downsample,
        base_mask=base_mask,
        coverage_mask=coverage_mask,
        prior_title=(
            "Shared all-example envelope without erosion"
            if args.prior_mode == "coverage-union"
            else (
                "Shared all-example envelope after upper-left erosion"
                if args.prior_mode == "coverage-eroded"
                else "Final Rv prior (P >= 0.20)"
            )
        ),
        placement=args.placement,
    )
    if args.prior_mode in {"coverage-union", "coverage-eroded"}:
        adapted = build_shared_adaptive_masks(
            results, prior_mask, adaptation_reference
        )
    else:
        adapted = build_individual_adaptive_masks(
            results,
            base_mask,
            seed,
            args.downsample,
            support_distance_pixels=256.0,
        )
    save_individual_adaptation_qc(
        output_dir,
        results,
        adapted,
        adaptation_reference,
        (
            "All-example coverage used directly (cyan/green)"
            if args.prior_mode == "coverage-union"
            else (
                "All-example coverage (cyan), shared eroded prior (green)"
                if args.prior_mode == "coverage-eroded"
                else "Fixed P>=0.2 prior (cyan), expansion-only mask (green)"
            )
        ),
        placement=args.placement,
    )

    full_shape = (full_height, full_width)
    probability_full = resize(
        probability,
        full_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    prior_full = resize(
        prior_mask.astype(np.uint8),
        full_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.uint8)
    prefix = args.placement.lower()
    tiff.imwrite(
        output_dir / (prefix + "_brain_prior_probability_float32.tif"),
        probability_full,
    )
    tiff.imwrite(
        output_dir / (prefix + "_brain_prior_mask_uint8.tif"),
        prior_full * 255,
    )
    base_full = resize(
        base_mask.astype(np.uint8),
        full_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.uint8)
    coverage_full = resize(
        coverage_mask.astype(np.uint8),
        full_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.uint8)
    tiff.imwrite(
        output_dir / (prefix + "_brain_prior_base_uint8.tif"), base_full * 255
    )
    tiff.imwrite(
        output_dir / (prefix + "_brain_prior_coverage_uint8.tif"),
        coverage_full * 255,
    )
    adapted_dir = output_dir / "individual_adapted_masks"
    adapted_dir.mkdir(parents=True, exist_ok=True)
    adaptation_rows = []
    for item in adapted:
        adapted_full = resize(
            item["mask"].astype(np.uint8),
            full_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.uint8)
        tiff.imwrite(
            adapted_dir / (item["dataset"] + "_adaptive_mask_uint8.tif"),
            adapted_full * 255,
        )
        adaptation_rows.append(
            {key: value for key, value in item.items() if key not in {"mask", "protected"}}
        )
    with (output_dir / "individual_adaptation_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(adaptation_rows[0]))
        writer.writeheader()
        writer.writerows(adaptation_rows)
    tiff.imwrite(
        output_dir / (prefix + "_median_aligned_projection_float32.tif"),
        median_projection.astype(np.float32),
    )

    rows = []
    for result in results:
        rows.append(
            {
                key: value for key, value in result.items()
                if key not in {"normalized", "valid", "mask"}
            }
        )
    with (output_dir / "individual_mask_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "placement": args.placement,
        "excluded": ["MAY08" + placement_suffix],
        "datasets": [result["dataset"] for result in results],
        "dataset_count": len(results),
        "downsample": args.downsample,
        "seed_radius_pixels": args.seed_radius,
        "probability_threshold": args.probability_threshold,
        "wall_exclusion_pixels": args.wall_exclusion,
        "adaptation_mode": args.prior_mode,
        "directional_erosion_pixels": (
            args.directional_erosion
            if args.prior_mode == "coverage-eroded"
            else 0.0
        ),
        "canonical_full_shape": list(full_shape),
        "prior_fraction": float(prior_mask.mean()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

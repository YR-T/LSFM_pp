"""Plot repeated directional erosion of an existing canonical prior."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import tifffile as tiff

from build_brain_mask_prior import erode_toward_upper_left


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--step-pixels", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    projection = tiff.imread(
        str(args.input_dir / "rv_median_aligned_projection_float32.tif")
    )
    coverage_full = tiff.imread(
        str(args.input_dir / "rv_brain_prior_coverage_uint8.tif")
    ) > 0
    coverage = resize(
        coverage_full.astype(np.uint8),
        projection.shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(bool)
    downsample = coverage_full.shape[0] / projection.shape[0]
    step_lowres = max(1, int(round(args.step_pixels / downsample)))

    masks = [coverage]
    for _ in range(args.iterations):
        masks.append(erode_toward_upper_left(masks[-1], step_lowres))

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), dpi=180)
    axes = axes.reshape(-1)
    initial_area = max(int(coverage.sum()), 1)
    for iteration, (axis, mask) in enumerate(zip(axes, masks)):
        axis.imshow(projection, cmap="gray", vmin=0, vmax=1)
        axis.contour(coverage, levels=[0.5], colors="cyan", linewidths=0.6)
        axis.contour(mask, levels=[0.5], colors="lime", linewidths=1.0)
        removed = 1.0 - float(mask.sum()) / initial_area
        axis.set_title(
            "{} x {} px | removed {:.1%}".format(
                iteration, args.step_pixels, removed
            )
        )
        axis.set_axis_off()
    for axis in axes[len(masks):]:
        axis.set_axis_off()
    fig.suptitle(
        "Repeated upper-left directional erosion: coverage (cyan), result (green)"
    )
    fig.tight_layout()
    output_path = args.input_dir / "rv_prior_repeated_erosion_sweep_qc.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()

from pathlib import Path
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, median_filter
from skimage.morphology import disk, opening
from skimage.exposure import rescale_intensity

def robust_percentile_clip(img, low=0.1, high=99.9):
    lo, hi = np.percentile(img, [low, high])
    img = np.clip(img, lo, hi)
    return img, lo, hi


def estimate_shading_field(img, sigma=80, eps=1e-6):
    """
    Very slow spatial background/illumination estimate.
    sigma should be much larger than c-Fos nuclear/blob radius.
    """
    img_float = img.astype(np.float32)
    smooth = gaussian_filter(img_float, sigma=sigma)
    smooth = np.maximum(smooth, eps)
    return smooth


def flatfield_like_correction(img, sigma=120):
    """
    Retrospective shading correction:
    divide by low-frequency field and preserve median intensity scale.
    """
    img_float = img.astype(np.float32)
    field = estimate_shading_field(img_float, sigma=sigma)
    corrected = img_float / field * np.median(field)
    corrected[corrected < 0] = 0
    return corrected.astype(np.float32), field.astype(np.float32)


def simple_destripe(img, axis=0, sigma=20, strength=1.0):
    """
    Simple additive stripe correction.

    axis=0: correct column-wise stripes, i.e. vertical stripes.
    axis=1: correct row-wise stripes, i.e. horizontal stripes.

    This estimates a stripe profile from median intensity along one axis,
    smooths it, and subtracts the residual profile.
    """
    img_float = img.astype(np.float32)

    if axis == 0:
        profile = np.median(img_float, axis=0)  # per column
        smooth_profile = gaussian_filter(profile, sigma=sigma)
        stripe_profile = profile - smooth_profile
        corrected = img_float - strength * stripe_profile[None, :]
    elif axis == 1:
        profile = np.median(img_float, axis=1)  # per row
        smooth_profile = gaussian_filter(profile, sigma=sigma)
        stripe_profile = profile - smooth_profile
        corrected = img_float - strength * stripe_profile[:, None]
    else:
        raise ValueError("axis must be 0 or 1")

    corrected[corrected < 0] = 0
    return corrected.astype(np.float32), stripe_profile.astype(np.float32)


def subtract_background_morphology(img, radius=40):
    """
    Morphological opening background subtraction.
    radius should be much larger than expected c-Fos blob radius.
    """
    img_float = img.astype(np.float32)
    bg = opening(img_float, footprint=disk(radius))
    sub = img_float - bg
    sub[sub < 0] = 0
    return sub.astype(np.float32), bg.astype(np.float32)


def save_qc_panel(out_png, raw, ffc, destriped, bg_sub, field=None, bg=None):
    imgs = [raw, ffc, destriped, bg_sub]
    titles = ["raw", "FFC/shading corrected", "destriped", "background subtracted"]

    if field is not None:
        imgs.append(field)
        titles.append("estimated shading field")
    if bg is not None:
        imgs.append(bg)
        titles.append("estimated background")

    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))

    if n == 1:
        axes = [axes]

    for ax, im, title in zip(axes, imgs, titles):
        disp, lo, hi = robust_percentile_clip(im, 0.5, 99.7)
        ax.imshow(disp, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def preprocess_one_image(
    in_tif,
    out_dir,
    shading_sigma=120,
    stripe_axis=0,
    stripe_sigma=30,
    stripe_strength=1.0,
    bg_radius=40,
):
    in_tif = Path(in_tif)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = tiff.imread(in_tif).astype(np.float32)

    # 1. crude offset correction: subtract very low percentile
    offset = np.percentile(raw, 0.05)
    raw0 = raw - offset
    raw0[raw0 < 0] = 0

    # 2. retrospective flat/shading correction
    ffc, field = flatfield_like_correction(raw0, sigma=shading_sigma)

#     3. improved destriping
#    destriped, stripe_field = local_horizontal_destripe(
#    ffc,
#    block_width=64,
#    y_smooth=3,
#    x_smooth=1,
#    strength=0.8,
#    presmooth_size=5,
#    )
    destriped = ffc
    # 4. background subtraction
    bg_sub, bg = subtract_background_morphology(destriped, radius=bg_radius)

    # save outputs
    stem = in_tif.stem
    tiff.imwrite(out_dir / f"{stem}_preprocessed.tif", bg_sub.astype(np.float32))
    tiff.imwrite(out_dir / f"{stem}_ffc.tif", ffc.astype(np.float32))
    tiff.imwrite(out_dir / f"{stem}_destriped.tif", destriped.astype(np.float32))

    # np.save(out_dir / f"{stem}_stripe_field.npy", stripe_field)

    save_qc_panel(
        out_dir / f"{stem}_qc.png",
        raw=raw0,
        ffc=ffc,
        destriped=destriped,
        bg_sub=bg_sub,
        field=field,
        bg=bg,
    )

    return {
        "input": str(in_tif),
        "output": str(out_dir / f"{stem}_preprocessed.tif"),
        "offset": float(offset),
        "shading_sigma": shading_sigma,
        "stripe_axis": stripe_axis,
        "stripe_sigma": stripe_sigma,
        "stripe_strength": stripe_strength,
        "bg_radius": bg_radius,
    }

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import interp1d

def local_horizontal_destripe(
    img,
    block_width=64,
    y_smooth=4,
    x_smooth=1,
    strength=1.0,
    presmooth_size=5,
):
    """
    Remove locally varying, nearly-horizontal stripe artifacts
    without rotating/resampling the image.

    img: 2D image
    block_width: width of local x-blocks used to estimate row profiles
    y_smooth: smoothing of row profiles
    x_smooth: smoothing across neighboring blocks
    strength: subtraction strength
    presmooth_size: median filter to reduce influence of bright spots
    """
    img = img.astype(np.float32)
    h, w = img.shape

    # suppress bright small blobs before estimating stripe profile
    work = median_filter(img, size=presmooth_size)

    centers = []
    profiles = []

    for x0 in range(0, w, block_width):
        x1 = min(w, x0 + block_width)
        block = work[:, x0:x1]

        # row-wise robust profile within this x block
        prof = np.median(block, axis=1)

        # remove slow y background, keep band-like component
        slow = gaussian_filter(prof, sigma=max(y_smooth * 4, 12))
        stripe_prof = prof - slow

        # smooth stripe profile along y
        stripe_prof = gaussian_filter(stripe_prof, sigma=y_smooth)

        centers.append((x0 + x1 - 1) / 2)
        profiles.append(stripe_prof)

    centers = np.asarray(centers)
    profiles = np.asarray(profiles)  # shape: n_blocks x h

    # smooth across x-blocks
    if profiles.shape[0] > 1 and x_smooth > 0:
        profiles = gaussian_filter(profiles, sigma=(x_smooth, 0))

    # interpolate stripe profile for every x
    xs = np.arange(w)
    f = interp1d(
        centers,
        profiles,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=(profiles[0], profiles[-1]),
    )
    stripe_profiles_x = f(xs)  # shape: w x h

    stripe_field = stripe_profiles_x.T  # h x w

    corrected = img - strength * stripe_field
    corrected[corrected < 0] = 0

    return corrected.astype(np.float32), stripe_field.astype(np.float32)

# Example
# preprocess_one_image(
#     "sample_slice.tif",
#     "preprocess_test",
#     shading_sigma=120,
#     stripe_axis=0,      # 0 if vertical stripes; 1 if horizontal stripes
#     stripe_sigma=30,
#     stripe_strength=0.8,
#     bg_radius=40,
# )
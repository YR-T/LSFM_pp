from pathlib import Path
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import pystripe

from scipy.ndimage import gaussian_filter
from skimage.morphology import disk, opening


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

def subtract_background_morphology(img, radius=40):
    """
    Morphological opening background subtraction.
    radius should be much larger than expected c-Fos blob radius.
    """
    img_float = img.astype(np.float32)
    selem = disk(radius)
    try:
        bg = opening(img_float, footprint=selem)
    except TypeError:
        bg = opening(img_float, selem=selem)
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
    stripe_sigma=(128, 256),
    stripe_level=7,
    stripe_wavelet='db2',
    bg_radius=20,
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

# filter a single image
    destriped = pystripe.filter_streaks(ffc, sigma=stripe_sigma, level=stripe_level, wavelet=stripe_wavelet)

#     3. improved destriping
#    destriped, stripe_field = local_horizontal_destripe(
#    ffc,
#    block_width=64,
#    y_smooth=3,
#    x_smooth=1,
#    strength=0.8,
#    presmooth_size=5,
#    )
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
        "stripe_sigma": stripe_sigma,
        "stripe_level": stripe_level,
        "stripe_wavelet": stripe_wavelet,
        "bg_radius": bg_radius,
    }

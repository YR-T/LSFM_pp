"""Combine per-dataset QC PNG files into one multi-page PDF."""

from argparse import ArgumentParser
from pathlib import Path

from PIL import Image


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_pdf", type=Path)
    parser.add_argument("--pattern", default="mask_and_spots_qc.png")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    image_paths = sorted(
        path for path in args.input_root.glob("*/{}".format(args.pattern))
        if path.is_file()
    )
    if not image_paths:
        raise FileNotFoundError(
            "No {} files below {}".format(args.pattern, args.input_root)
        )

    pages = []
    try:
        for index, image_path in enumerate(image_paths, start=1):
            with Image.open(image_path) as image:
                pages.append(image.convert("RGB"))
            print("[{}/{}] {}".format(index, len(image_paths), image_path.parent.name))

        args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_pdf.with_suffix(".tmp.pdf")
        pages[0].save(
            temporary,
            "PDF",
            save_all=True,
            append_images=pages[1:],
            resolution=float(args.dpi),
            quality=90,
        )
        temporary.replace(args.output_pdf)
    finally:
        for page in pages:
            page.close()

    print("Saved {} pages to {}".format(len(image_paths), args.output_pdf))


if __name__ == "__main__":
    main()

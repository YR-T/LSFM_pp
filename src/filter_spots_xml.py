from argparse import ArgumentParser
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


DEFAULT_THRESHOLDS = {
    "SNR_CH1": 0.27,
    "TOTAL_INTENSITY_CH1": 492.0,
    "CONTRAST_CH1": 0.12,
    "MAX_INTENSITY_CH1": 280.0,
}


def keep_spot(attrs):
    return (
        float(attrs["SNR_CH1"]) > DEFAULT_THRESHOLDS["SNR_CH1"]
        and float(attrs["TOTAL_INTENSITY_CH1"]) > DEFAULT_THRESHOLDS["TOTAL_INTENSITY_CH1"]
        and float(attrs["CONTRAST_CH1"]) > DEFAULT_THRESHOLDS["CONTRAST_CH1"]
        and float(attrs["MAX_INTENSITY_CH1"]) < DEFAULT_THRESHOLDS["MAX_INTENSITY_CH1"]
    )


def write_filtered_spots_xml(input_xml, output_xml):
    count = 0
    with output_xml.open("w", encoding="utf-8", newline="\n") as out:
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        out.write("<Spots>\n")

        for _, elem in ET.iterparse(input_xml, events=("end",)):
            if elem.tag == "Spot" and keep_spot(elem.attrib):
                attrs = " ".join(f'{key}="{value}"' for key, value in elem.attrib.items())
                out.write(f"  <Spot {attrs} />\n")
                count += 1
            elem.clear()

        out.write("</Spots>\n")

    return count


def selected_ids_from_pickle(pkl_path):
    df = pd.read_pickle(pkl_path)
    mask = (
        (df["SNR_CH1"] > DEFAULT_THRESHOLDS["SNR_CH1"])
        & (df["TOTAL_INTENSITY_CH1"] > DEFAULT_THRESHOLDS["TOTAL_INTENSITY_CH1"])
        & (df["CONTRAST_CH1"] > DEFAULT_THRESHOLDS["CONTRAST_CH1"])
        & (df["MAX_INTENSITY_CH1"] < DEFAULT_THRESHOLDS["MAX_INTENSITY_CH1"])
    )
    return set(df.loc[mask, "ID"].astype(str))


def write_spots_by_id(input_xml, output_xml, selected_ids):
    count = 0
    with output_xml.open("w", encoding="utf-8", newline="\n") as out:
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        out.write("<Spots>\n")

        for _, elem in ET.iterparse(input_xml, events=("end",)):
            if elem.tag == "Spot" and elem.attrib["ID"] in selected_ids:
                attrs = " ".join(f'{key}="{value}"' for key, value in elem.attrib.items())
                out.write(f"  <Spot {attrs} />\n")
                count += 1
            elem.clear()

        out.write("</Spots>\n")

    return count


def main():
    parser = ArgumentParser(description="Write filtered TrackMate Spot elements to XML.")
    parser.add_argument("xml", type=Path, help="Input XML file.")
    parser.add_argument(
        "--spots-pkl",
        type=Path,
        help="Optional Spot DataFrame pickle. When provided, filtering is done on the DataFrame and original XML Spot attributes are written by matching ID.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output XML path. Defaults to '<xml_stem>_filtered_spots.xml'.",
    )
    args = parser.parse_args()

    output_xml = args.output or args.xml.with_name(f"{args.xml.stem}_filtered_spots.xml")
    if args.spots_pkl:
        selected_ids = selected_ids_from_pickle(args.spots_pkl)
        count = write_spots_by_id(args.xml, output_xml, selected_ids)
    else:
        count = write_filtered_spots_xml(args.xml, output_xml)
    print(f"Saved {count:,} filtered spots to {output_xml}")


if __name__ == "__main__":
    main()

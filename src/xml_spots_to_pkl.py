from argparse import ArgumentParser
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


def spots_xml_to_dataframe(xml_path):
    rows = []

    for _, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag == "Spot":
            rows.append(dict(elem.attrib))
            elem.clear()

    df = pd.DataFrame(rows)

    for column in df.columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().all():
            df[column] = numeric

    return df


def main():
    parser = ArgumentParser(description="Extract TrackMate <Spot> elements to a pandas pickle.")
    parser.add_argument("xml", type=Path, help="Input XML file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .pkl path. Defaults to '<xml_stem>_spots.pkl' next to the XML.",
    )
    args = parser.parse_args()

    xml_path = args.xml
    output_path = args.output or xml_path.with_name(f"{xml_path.stem}_spots.pkl")

    df = spots_xml_to_dataframe(xml_path)
    df.to_pickle(output_path)

    print(f"Saved {len(df):,} spots x {len(df.columns):,} columns to {output_path}")
    print(df.dtypes)


if __name__ == "__main__":
    main()

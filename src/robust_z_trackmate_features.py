"""Add block-normalized robust Z features to Q-filtered TrackMate Spots."""

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from xml.sax.saxutils import quoteattr

import numpy as np


QUALITY_THRESHOLD = 150.0
ROBUST_SCALE_FACTOR = 1.4826
FEATURE_TRANSFORM = "log1p_contrast_snr_floor0"
FEATURES = (
    ("QUALITY", "QUALITY"),
    ("MIN_INTENSITY_CH1", "MIN_INTENSITY_CH1"),
    ("MEAN_INTENSITY_CH1", "MEAN_INTENSITY_CH1"),
    ("MEDIAN_INTENSITY_CH1", "MEDIAN_INTENSITY_CH1"),
    ("MAX_INTENSITY_CH1", "MAX_INTENSITY_CH1"),
    ("TOTAL_INTENSITY_CH1", "TOTAL_INTENSITY_CH1"),
    ("STD_INTENSITY_CH1", "STD_INTENSITY_CH1"),
    ("CV", None),
    ("CONTRAST_CH1", "CONTRAST_CH1"),
    ("SNR_CH1", "SNR_CH1"),
)


def block_name(dataset):
    return dataset.split("_FOS_", 1)[0]


def feature_values(attrs):
    mean = float(attrs["MEAN_INTENSITY_CH1"])
    sd = float(attrs["STD_INTENSITY_CH1"])
    values = []
    for feature, source_attribute in FEATURES:
        if feature == "CV":
            values.append(sd / mean if mean != 0 else float("inf"))
        else:
            values.append(float(attrs[source_attribute]))
    return np.asarray(values, dtype=np.float64)


def transformed_feature_values(attrs):
    values = feature_values(attrs)
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite feature value found.")
    values[8] = max(values[8], 0.0)  # Contrast
    values[9] = max(values[9], 0.0)  # SNR
    if np.any(values < 0):
        raise ValueError("Negative value found outside Contrast/SNR.")
    return np.log1p(values)


def read_q_filtered_matrix(xml_path, expected_spots):
    matrix = np.empty((int(expected_spots), len(FEATURES)), dtype=np.float64)
    index = 0
    for _, element in ET.iterparse(str(xml_path), events=("end",)):
        if element.tag == "Spot":
            quality = float(element.attrib["QUALITY"])
            if quality >= QUALITY_THRESHOLD:
                matrix[index] = transformed_feature_values(element.attrib)
                index += 1
        element.clear()
    return matrix[:index].copy()


def serialize_spot(attrs):
    return "  <Spot {} />\n".format(
        " ".join(
            "{}={}".format(key, quoteattr(str(value)))
            for key, value in attrs.items()
        )
    )


def write_robust_xml(source_xml, output_xml, block, medians, scales):
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_xml.with_suffix(".tmp")
    kept = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        output.write(
            (
                '<spots block={} quality_threshold={} feature_transform={} '
                'robust_z_method={}>\n'
            ).format(
                quoteattr(block),
                quoteattr("{:g}".format(QUALITY_THRESHOLD)),
                quoteattr(FEATURE_TRANSFORM),
                quoteattr("median_mad"),
            )
        )
        for _, element in ET.iterparse(str(source_xml), events=("end",)):
            if element.tag == "Spot":
                attrs = dict(element.attrib)
                quality = float(attrs["QUALITY"])
                if quality >= QUALITY_THRESHOLD:
                    raw_values = feature_values(attrs)
                    transformed_values = transformed_feature_values(attrs)
                    attrs["CV"] = "{:.9g}".format(raw_values[7])
                    robust_values = (transformed_values - medians) / scales
                    for index, (feature, _) in enumerate(FEATURES):
                        attrs["ROBUST_Z_" + feature] = "{:.9g}".format(
                            robust_values[index]
                        )
                    output.write(serialize_spot(attrs))
                    kept += 1
            element.clear()
        output.write("</spots>\n")
    temporary.replace(output_xml)
    return kept


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    project_root = Path(__file__).resolve().parent.parent
    default_root = (
        project_root
        / "outputs"
        / "TMoptimization"
        / "r2p5_median_off_slices_040_080"
    )
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=default_root)
    args = parser.parse_args()

    source_summary = args.result_root / "detection_summary.csv"
    if not source_summary.is_file():
        raise FileNotFoundError(source_summary)
    with source_summary.open("r", encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    if len(source_rows) != 80:
        raise RuntimeError("Expected 80 source rows, found {}".format(len(source_rows)))

    rows_by_block = {}
    for row in source_rows:
        row["slice"] = int(row["slice"])
        row["spots"] = int(row["spots"])
        row["block"] = block_name(row["dataset"])
        stored_xml = Path(row["output_xml"])
        row["source_xml"] = (
            args.result_root / "raw_xml" / row["dataset"] / stored_xml.name
        )
        if not row["source_xml"].is_file():
            raise FileNotFoundError(row["source_xml"])
        rows_by_block.setdefault(row["block"], []).append(row)

    if len(rows_by_block) != 20:
        raise RuntimeError("Expected 20 L/R blocks, found {}".format(len(rows_by_block)))
    for block, rows in rows_by_block.items():
        conditions = {(row["orientation"], row["slice"]) for row in rows}
        if len(rows) != 4 or conditions != {
            ("d", 40),
            ("d", 80),
            ("v", 40),
            ("v", 80),
        }:
            raise RuntimeError("{} does not contain d/v x slice 40/80.".format(block))

    output_dir = args.result_root / "robust_z_log1p_floor0_q150_xml"
    state_path = args.result_root / "robust_z_log1p_floor0_status.json"
    state = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "calculating_block_statistics",
        "quality_threshold": QUALITY_THRESHOLD,
        "normalization": "median_mad",
        "feature_transform": FEATURE_TRANSFORM,
        "robust_scale_factor": ROBUST_SCALE_FACTOR,
        "features": [feature for feature, _ in FEATURES],
        "blocks_total": len(rows_by_block),
        "blocks_completed": 0,
        "xml_total": len(source_rows),
        "xml_completed": 0,
    }
    write_json(state_path, state)

    statistics = {}
    statistics_rows = []
    for block_index, block in enumerate(sorted(rows_by_block), start=1):
        matrices = []
        for row in rows_by_block[block]:
            matrices.append(
                read_q_filtered_matrix(row["source_xml"], row["spots"])
            )
        block_matrix = np.concatenate(matrices, axis=0)
        if len(block_matrix) == 0:
            raise RuntimeError("{} has no Q >= 150 Spots.".format(block))
        medians = np.median(block_matrix, axis=0)
        mads = np.median(np.abs(block_matrix - medians), axis=0)
        scales = ROBUST_SCALE_FACTOR * mads
        zero_scale = [
            FEATURES[index][0]
            for index, scale in enumerate(scales)
            if not np.isfinite(scale) or scale <= 0
        ]
        if zero_scale:
            raise RuntimeError(
                "{} has invalid MAD scale for {}".format(block, zero_scale)
            )
        statistics[block] = {"medians": medians, "mads": mads, "scales": scales}
        for feature_index, (feature, _) in enumerate(FEATURES):
            statistics_rows.append(
                {
                    "block": block,
                    "feature": feature,
                    "q150_spots": len(block_matrix),
                    "median": "{:.12g}".format(medians[feature_index]),
                    "mad": "{:.12g}".format(mads[feature_index]),
                    "robust_scale": "{:.12g}".format(scales[feature_index]),
                }
            )
        state["blocks_completed"] = block_index
        write_json(state_path, state)
        print(
            "[BLOCK {:02d}/20] {} Q>=150 spots={:,}".format(
                block_index, block, len(block_matrix)
            ),
            flush=True,
        )

    write_csv(
        args.result_root / "robust_z_log1p_floor0_block_statistics.csv",
        statistics_rows,
        ["block", "feature", "q150_spots", "median", "mad", "robust_scale"],
    )
    statistics_json = {
        block: {
            feature: {
                "median": float(values["medians"][index]),
                "mad": float(values["mads"][index]),
                "robust_scale": float(values["scales"][index]),
            }
            for index, (feature, _) in enumerate(FEATURES)
        }
        for block, values in statistics.items()
    }
    write_json(
        args.result_root / "robust_z_log1p_floor0_block_statistics.json",
        {
            "quality_threshold": QUALITY_THRESHOLD,
            "feature_transform": FEATURE_TRANSFORM,
            "normalization": "median_mad",
            "robust_scale_factor": ROBUST_SCALE_FACTOR,
            "blocks": statistics_json,
        },
    )

    state["status"] = "writing_robust_z_xml"
    write_json(state_path, state)
    output_rows = []
    for xml_index, row in enumerate(
        sorted(source_rows, key=lambda value: (value["dataset"], value["slice"])),
        start=1,
    ):
        output_xml = (
            output_dir
            / row["dataset"]
            / (
                row["source_xml"].stem
                + "_q150_log1p_floor0_robust_z.xml"
            )
        )
        stats = statistics[row["block"]]
        kept = write_robust_xml(
            row["source_xml"],
            output_xml,
            row["block"],
            stats["medians"],
            stats["scales"],
        )
        output_rows.append(
            {
                "block": row["block"],
                "dataset": row["dataset"],
                "orientation": row["orientation"],
                "slice": row["slice"],
                "source_spots_q0": row["spots"],
                "spots_q150": kept,
                "source_xml": str(row["source_xml"]),
                "output_xml": str(output_xml),
                "output_bytes": output_xml.stat().st_size,
            }
        )
        state["xml_completed"] = xml_index
        write_json(state_path, state)
        print(
            "[XML {:02d}/80] {} slice={:03d} Q>=150 spots={:,}".format(
                xml_index, row["dataset"], row["slice"], kept
            ),
            flush=True,
        )

    write_csv(
        args.result_root / "robust_z_log1p_floor0_summary.csv",
        output_rows,
        [
            "block",
            "dataset",
            "orientation",
            "slice",
            "source_spots_q0",
            "spots_q150",
            "source_xml",
            "output_xml",
            "output_bytes",
        ],
    )
    state["status"] = "complete"
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["total_q150_spots"] = sum(row["spots_q150"] for row in output_rows)
    write_json(state_path, state)
    print(
        "COMPLETE: 80 XML files, {:,} Q>=150 Spots".format(
            state["total_q150_spots"]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

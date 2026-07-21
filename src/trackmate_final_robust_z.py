"""Create hemisphere-wise log1p Robust Z XML and thresholded condition XML."""

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
ROBUST_BOUNDS = {
    "CONTRAST_CH1": (0.0, 10.0),
    "SNR_CH1": (0.0, 10.0),
    "STD_INTENSITY_CH1": (-2.0, 6.0),
    "CV": (-2.0, 5.0),
    "MEAN_INTENSITY_CH1": (-3.0, 6.0),
    "MIN_INTENSITY_CH1": (-3.0, 3.5),
}
FEATURE_INDEX = {
    feature: index for index, (feature, _) in enumerate(FEATURES)
}


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def current_path(project_root, path_string):
    path = Path(path_string)
    parts = list(path.parts)
    if "outputs" in parts:
        return project_root.joinpath(*parts[parts.index("outputs"):])
    return path


def raw_feature_values(attrs):
    mean = float(attrs["MEAN_INTENSITY_CH1"])
    sd = float(attrs["STD_INTENSITY_CH1"])
    values = []
    for feature, source in FEATURES:
        if feature == "CV":
            values.append(sd / mean if mean != 0 else float("inf"))
        else:
            values.append(float(attrs[source]))
    values = np.asarray(values, dtype=np.float64)
    for feature in ("CONTRAST_CH1", "SNR_CH1"):
        index = FEATURE_INDEX[feature]
        if not np.isfinite(values[index]):
            values[index] = 0.0
    if not np.all(np.isfinite(values)):
        invalid = [
            FEATURES[index][0]
            for index in np.flatnonzero(~np.isfinite(values))
        ]
        raise ValueError(
            "Non-finite raw feature values outside Contrast/SNR: {}".format(invalid)
        )
    return values


def transformed_values(attrs):
    values = raw_feature_values(attrs)
    values[FEATURE_INDEX["CONTRAST_CH1"]] = max(
        values[FEATURE_INDEX["CONTRAST_CH1"]], 0.0
    )
    values[FEATURE_INDEX["SNR_CH1"]] = max(
        values[FEATURE_INDEX["SNR_CH1"]], 0.0
    )
    if np.any(values < 0):
        negative = [
            FEATURES[index][0]
            for index in np.flatnonzero(values < 0)
        ]
        raise ValueError("Negative values outside Contrast/SNR: {}".format(negative))
    return np.log1p(values)


def fill_matrix(xml_path, matrix, offset):
    index = int(offset)
    for _, element in ET.iterparse(str(xml_path), events=("end",)):
        if element.tag == "Spot":
            quality = float(element.attrib["QUALITY"])
            if quality >= QUALITY_THRESHOLD:
                if index >= len(matrix):
                    raise RuntimeError("More spots than summary in {}".format(xml_path))
                matrix[index] = transformed_values(element.attrib)
                index += 1
        element.clear()
    return index


def serialize_spot(attrs, indent="  "):
    return "{}<Spot {} />\n".format(
        indent,
        " ".join(
            "{}={}".format(key, quoteattr(str(value)))
            for key, value in attrs.items()
        ),
    )


def passes_robust_bounds(robust_values):
    for feature, (lower, upper) in ROBUST_BOUNDS.items():
        value = robust_values[FEATURE_INDEX[feature]]
        if not (value > lower and value < upper):
            return False
    return True


def open_xml(path, root_line):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    stream = temporary.open("w", encoding="utf-8", newline="\n")
    stream.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    stream.write(root_line + "\n")
    return stream, temporary


def finish_xml(stream, temporary, output_path):
    stream.write("</spots>\n")
    stream.close()
    temporary.replace(output_path)


def write_slice(
    source_xml,
    robust_xml,
    filtered_xml,
    block,
    dataset,
    slice_number,
    medians,
    scales,
    combined,
    next_global_id,
):
    robust_root = (
        '<spots block={} dataset={} slice={} quality_threshold={} '
        'feature_transform={} robust_z_method={}>'
    ).format(
        quoteattr(block),
        quoteattr(dataset),
        quoteattr(str(slice_number)),
        quoteattr("{:g}".format(QUALITY_THRESHOLD)),
        quoteattr(FEATURE_TRANSFORM),
        quoteattr("median_mad"),
    )
    robust_out, robust_tmp = open_xml(robust_xml, robust_root)
    filtered_out, filtered_tmp = open_xml(
        filtered_xml,
        '<spots dataset={} slice={} threshold_set={}>'.format(
            quoteattr(dataset),
            quoteattr(str(slice_number)),
            quoteattr("initial_common_20260721"),
        ),
    )
    q150_count = 0
    selected_count = 0
    try:
        for _, element in ET.iterparse(str(source_xml), events=("end",)):
            if element.tag == "Spot":
                attrs = dict(element.attrib)
                if float(attrs["QUALITY"]) >= QUALITY_THRESHOLD:
                    raw_values = raw_feature_values(attrs)
                    robust_values = (
                        transformed_values(attrs) - medians
                    ) / scales
                    attrs["CV"] = "{:.9g}".format(
                        raw_values[FEATURE_INDEX["CV"]]
                    )
                    for feature_index, (feature, _) in enumerate(FEATURES):
                        attrs["ROBUST_Z_" + feature] = "{:.9g}".format(
                            robust_values[feature_index]
                        )
                    robust_out.write(serialize_spot(attrs))
                    q150_count += 1
                    if passes_robust_bounds(robust_values):
                        selected_attrs = dict(attrs)
                        selected_attrs["POSITION_Z"] = str(slice_number - 1)
                        selected_attrs["ID"] = str(next_global_id)
                        selected_attrs["name"] = "ID{}".format(next_global_id)
                        line = serialize_spot(selected_attrs)
                        filtered_out.write(line)
                        combined.write(line)
                        selected_count += 1
                        next_global_id += 1
            element.clear()
        finish_xml(robust_out, robust_tmp, robust_xml)
        finish_xml(filtered_out, filtered_tmp, filtered_xml)
    except Exception:
        robust_out.close()
        filtered_out.close()
        raise
    return q150_count, selected_count, next_global_id


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=project_root / "outputs" / "TrackMate_final_r2p5_q150",
    )
    parser.add_argument(
        "--block",
        help="Process only one hemisphere block and write partial summaries.",
    )
    args = parser.parse_args()
    detection_summary = args.result_root / "detection_summary.csv"
    if not detection_summary.is_file():
        raise FileNotFoundError(detection_summary)
    with detection_summary.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 7200:
        raise RuntimeError("Expected 7,200 detection rows, found {}".format(len(rows)))
    for row in rows:
        row["slice"] = int(row["slice"])
        row["spots_q150"] = int(row["spots_q150"])
        row["q150_xml"] = current_path(project_root, row["q150_xml"])
        row["input_tif"] = current_path(project_root, row["input_tif"])

    rows_by_block = {}
    for row in rows:
        rows_by_block.setdefault(row["hemisphere"], []).append(row)
    if len(rows_by_block) != 20:
        raise RuntimeError("Expected 20 hemisphere blocks.")
    expected_grid = {
        (orientation, slice_number)
        for orientation in ("d", "v")
        for slice_number in range(1, 181)
    }
    for block, block_rows in rows_by_block.items():
        actual_grid = {
            (row["orientation"], row["slice"]) for row in block_rows
        }
        if len(block_rows) != 360 or actual_grid != expected_grid:
            raise RuntimeError("{} does not contain d/v x slices 1..180.".format(block))
    if args.block:
        if args.block not in rows_by_block:
            raise RuntimeError("Unknown block: {}".format(args.block))
        rows_by_block = {args.block: rows_by_block[args.block]}

    if args.block:
        state_path = (
            args.result_root
            / "partial_summaries"
            / (args.block + "_status.json")
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        state_path = args.result_root / "robust_z_status.json"
    state = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
        "quality_threshold": QUALITY_THRESHOLD,
        "feature_transform": FEATURE_TRANSFORM,
        "normalization": "median_mad",
        "robust_scale_factor": ROBUST_SCALE_FACTOR,
        "features": [feature for feature, _ in FEATURES],
        "robust_bounds_exclusive": ROBUST_BOUNDS,
        "blocks_total": len(rows_by_block),
        "blocks_completed": 0,
        "xml_total": len(rows_by_block) * 360,
        "xml_completed": 0,
    }
    write_json(state_path, state)
    if not args.block:
        write_json(
            args.result_root / "filter_thresholds_robust_z.json",
            {
                "quality_threshold_inclusive": QUALITY_THRESHOLD,
                "feature_transform": FEATURE_TRANSFORM,
                "normalization": "median_mad",
                "robust_scale_factor": ROBUST_SCALE_FACTOR,
                "bounds_exclusive": ROBUST_BOUNDS,
            },
        )

    statistics_rows = []
    output_rows = []
    condition_rows = []
    total_xml_completed = 0
    block_total = len(rows_by_block)
    for block_index, block in enumerate(sorted(rows_by_block), start=1):
        block_rows = sorted(
            rows_by_block[block],
            key=lambda row: (row["orientation"], row["slice"]),
        )
        expected_count = sum(row["spots_q150"] for row in block_rows)
        matrix = np.empty((expected_count, len(FEATURES)), dtype=np.float64)
        offset = 0
        for row in block_rows:
            offset = fill_matrix(row["q150_xml"], matrix, offset)
        if offset != expected_count:
            raise RuntimeError(
                "{} count mismatch: summary {}, parsed {}".format(
                    block, expected_count, offset
                )
            )
        medians = np.median(matrix, axis=0)
        mads = np.median(np.abs(matrix - medians), axis=0)
        scales = ROBUST_SCALE_FACTOR * mads
        invalid = [
            FEATURES[index][0]
            for index in range(len(FEATURES))
            if not np.isfinite(scales[index]) or scales[index] <= 0
        ]
        if invalid:
            raise RuntimeError("{} invalid MAD scales: {}".format(block, invalid))
        for feature_index, (feature, _) in enumerate(FEATURES):
            statistics_rows.append(
                {
                    "block": block,
                    "feature": feature,
                    "q150_spots": expected_count,
                    "transformed_median": "{:.12g}".format(medians[feature_index]),
                    "transformed_mad": "{:.12g}".format(mads[feature_index]),
                    "robust_scale": "{:.12g}".format(scales[feature_index]),
                }
            )
        del matrix

        for orientation in ("d", "v"):
            dataset = "{}_FOS_{}".format(block, orientation)
            condition_output = (
                args.result_root
                / "filtered_by_condition"
                / "{}_{}_filtered_spots.xml".format(block, orientation)
            )
            combined, combined_tmp = open_xml(
                condition_output,
                '<spots dataset={} threshold_set={}>'.format(
                    quoteattr(dataset),
                    quoteattr("initial_common_20260721"),
                ),
            )
            condition_q150 = 0
            condition_selected = 0
            next_global_id = 0
            condition_source_rows = [
                row for row in block_rows if row["orientation"] == orientation
            ]
            try:
                for row in condition_source_rows:
                    robust_xml = (
                        args.result_root
                        / "robust_z_xml"
                        / dataset
                        / (
                            row["q150_xml"].stem
                            + "_log1p_robust_z.xml"
                        )
                    )
                    filtered_xml = (
                        args.result_root
                        / "filtered_by_slice"
                        / dataset
                        / (
                            row["q150_xml"].stem
                            + "_filtered_spots.xml"
                        )
                    )
                    q150_count, selected_count, next_global_id = write_slice(
                        row["q150_xml"],
                        robust_xml,
                        filtered_xml,
                        block,
                        dataset,
                        row["slice"],
                        medians,
                        scales,
                        combined,
                        next_global_id,
                    )
                    if q150_count != row["spots_q150"]:
                        raise RuntimeError(
                            "{} slice {} expected {}, wrote {}".format(
                                dataset,
                                row["slice"],
                                row["spots_q150"],
                                q150_count,
                            )
                        )
                    condition_q150 += q150_count
                    condition_selected += selected_count
                    output_rows.append(
                        {
                            "block": block,
                            "dataset": dataset,
                            "orientation": orientation,
                            "slice": row["slice"],
                            "input_tif": str(row["input_tif"]),
                            "q150_xml": str(row["q150_xml"]),
                            "robust_z_xml": str(robust_xml),
                            "filtered_slice_xml": str(filtered_xml),
                            "spots_q150": q150_count,
                            "spots_selected": selected_count,
                        }
                    )
                    total_xml_completed += 1
                    state["xml_completed"] = total_xml_completed
                    write_json(state_path, state)
                finish_xml(combined, combined_tmp, condition_output)
            except Exception:
                combined.close()
                raise
            condition_rows.append(
                {
                    "block": block,
                    "dataset": dataset,
                    "orientation": orientation,
                    "spots_q150": condition_q150,
                    "spots_selected": condition_selected,
                    "selected_fraction": (
                        condition_selected / float(condition_q150)
                        if condition_q150 else 0.0
                    ),
                    "output_xml": str(condition_output),
                }
            )
            print(
                "[BLOCK {:02d}/{}] {} {}: Q>=150 {:,}, selected {:,}".format(
                    block_index,
                    block_total,
                    block,
                    orientation,
                    condition_q150,
                    condition_selected,
                ),
                flush=True,
            )
        state["blocks_completed"] = block_index
        write_json(state_path, state)

    partial_prefix = (
        args.result_root / "partial_summaries" / args.block
        if args.block
        else args.result_root
    )
    if args.block:
        partial_prefix.parent.mkdir(parents=True, exist_ok=True)
        statistics_path = Path(str(partial_prefix) + "_statistics.csv")
        slice_summary_path = Path(str(partial_prefix) + "_slices.csv")
        condition_summary_path = Path(str(partial_prefix) + "_conditions.csv")
    else:
        statistics_path = args.result_root / "robust_z_block_statistics.csv"
        slice_summary_path = args.result_root / "robust_z_slice_summary.csv"
        condition_summary_path = args.result_root / "filtered_condition_summary.csv"
    write_csv(
        statistics_path,
        statistics_rows,
        [
            "block",
            "feature",
            "q150_spots",
            "transformed_median",
            "transformed_mad",
            "robust_scale",
        ],
    )
    write_csv(
        slice_summary_path,
        output_rows,
        [
            "block",
            "dataset",
            "orientation",
            "slice",
            "input_tif",
            "q150_xml",
            "robust_z_xml",
            "filtered_slice_xml",
            "spots_q150",
            "spots_selected",
        ],
    )
    write_csv(
        condition_summary_path,
        condition_rows,
        [
            "block",
            "dataset",
            "orientation",
            "spots_q150",
            "spots_selected",
            "selected_fraction",
            "output_xml",
        ],
    )
    state["status"] = "complete"
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["total_q150_spots"] = sum(row["spots_q150"] for row in output_rows)
    state["total_selected_spots"] = sum(
        row["spots_selected"] for row in output_rows
    )
    write_json(state_path, state)
    print(
        "COMPLETE: {:,} Robust Z XML; {:,} selected from {:,}".format(
            len(output_rows),
            state["total_selected_spots"],
            state["total_q150_spots"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from quality_model.features import extract_phy_features
from quality_model.scoring import score_frame, write_phy_probability


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score every cluster in one Phy output folder."
    )
    parser.add_argument(
        "phy_dir",
        type=Path,
        help="Path to the Phy output folder containing spike_clusters.npy",
    )
    parser.add_argument("model", type=Path, help="Path to a Stine-Score .joblib model")
    parser.add_argument(
        "--column-name", default="good_probability", help="Phy metadata column name"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output TSV path; defaults to cluster_<column-name>.tsv in the Phy folder",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output TSV."
    )
    args = parser.parse_args()

    if not args.column_name.replace("_", "").isalnum():
        raise ValueError("Column name may contain only letters, numbers, and underscores")

    phy_dir = args.phy_dir.resolve()
    output = args.output or phy_dir / f"cluster_{args.column_name}.tsv"
    artifact = joblib.load(args.model)
    frame = extract_phy_features(phy_dir, include_waveforms=False, all_clusters=True)
    probability = score_frame(artifact, frame)
    values = frame[["cluster_id"]].copy()
    values[args.column_name] = probability
    if values["cluster_id"].duplicated().any():
        raise ValueError("Duplicate cluster IDs found")
    write_phy_probability(output, values.sort_values("cluster_id"), overwrite=args.overwrite)
    print(f"Scored {len(values)} clusters.")
    print(f"Wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

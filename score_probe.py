#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from quality_model.features import extract_phy_features
from quality_model.scoring import raw_score_frame, score_frame, write_phy_probability


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
        "--overwrite", action="store_true", help="Replace an existing output TSV."
    )
    args = parser.parse_args()

    phy_dir = args.phy_dir.resolve()
    artifact = joblib.load(args.model)
    frame = extract_phy_features(phy_dir, include_waveforms=False, all_clusters=True)
    probability = score_frame(artifact, frame)
    raw_score = raw_score_frame(artifact, frame)
    values = frame[["cluster_id"]].copy()
    values["good_probability"] = probability
    values["good_unit_score"] = raw_score
    if values["cluster_id"].duplicated().any():
        raise ValueError("Duplicate cluster IDs found")
    calibrated_path = phy_dir / "cluster_good_probability.tsv"
    raw_score_path = phy_dir / "cluster_good_unit_score.tsv"
    write_phy_probability(
        calibrated_path,
        values[["cluster_id", "good_probability"]].sort_values("cluster_id"),
        overwrite=args.overwrite,
    )
    write_phy_probability(
        raw_score_path,
        values[["cluster_id", "good_unit_score"]].sort_values("cluster_id"),
        overwrite=args.overwrite,
    )
    print(f"Scored {len(values)} clusters.")
    print(f"Wrote {calibrated_path.resolve()}")
    print(f"Wrote {raw_score_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

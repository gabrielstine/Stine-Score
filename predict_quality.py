#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Score extracted unit features.")
    parser.add_argument("model", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("--output", type=Path, default=Path("quality_probabilities.csv"))
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    frame = pd.read_csv(args.features)
    columns = artifact["feature_columns"]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Feature file is missing model inputs: {sorted(missing)}")
    x = frame[columns].replace([np.inf, -np.inf], np.nan)
    raw = artifact["model"].predict_proba(x)[:, 1]
    probability = artifact["calibrator"].predict(raw)
    identifiers = [
        column
        for column in ["session", "stream", "region", "cluster_id", "label"]
        if column in frame.columns
    ]
    output = frame[identifiers].copy()
    output["good_probability"] = probability
    output["model_name"] = artifact["model_name"]
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} probabilities to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

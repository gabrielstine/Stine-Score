#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from quality_model.modeling import (
    MODEL_FACTORIES,
    feature_columns,
    fit_final_calibrated_model,
    nested_group_predictions,
    probability_metrics,
    reliability_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and compare session-validated unit-quality models."
    )
    parser.add_argument("features", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--random-state", type=int, default=20260727)
    args = parser.parse_args()

    frame = pd.read_csv(args.features)
    required = {"session", "region", "cluster_id", "label", "y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature file is missing columns: {sorted(missing)}")
    frame = frame[frame["label"].isin(["good", "mua", "noise"])].reset_index(drop=True)
    columns = feature_columns(frame)
    if not columns:
        raise ValueError("No numeric feature columns were found")
    x = frame[columns].replace([np.inf, -np.inf], np.nan)
    y = frame["y"].astype(int).to_numpy()
    groups = frame["session"].astype(str).to_numpy()
    if np.unique(groups).size < 3:
        raise ValueError("At least three sessions are required for nested grouped validation")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_columns = frame[["session", "stream", "region", "cluster_id", "label", "y"]].copy()
    all_metrics: dict[str, dict[str, float]] = {}

    for name, factory in MODEL_FACTORIES.items():
        print(f"Validating {name} with held-out sessions...", flush=True)
        raw, calibrated = nested_group_predictions(
            factory, x, y, groups, random_state=args.random_state
        )
        prediction_columns[f"{name}_raw_probability"] = raw
        prediction_columns[f"{name}_probability"] = calibrated
        metrics = probability_metrics(y, calibrated)
        metrics.update({f"raw_{key}": value for key, value in probability_metrics(y, raw).items()})
        all_metrics[name] = metrics
        reliability_table(y, calibrated).to_csv(
            output_dir / f"reliability_{name}.csv", index=False
        )

    prediction_columns.to_csv(output_dir / "held_out_session_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )

    # Brier score is the primary selection criterion because the requested output
    # is a calibrated personal-label probability.
    winner = min(all_metrics, key=lambda name: all_metrics[name]["brier_score"])
    print(f"Fitting final model: {winner}", flush=True)
    fitted = fit_final_calibrated_model(
        MODEL_FACTORIES[winner], x, y, groups, random_state=args.random_state
    )
    artifact = {
        **fitted,
        "model_name": winner,
        "feature_columns": columns,
        "training_sessions": sorted(np.unique(groups).tolist()),
        "training_rows": int(len(frame)),
        "positive_fraction": float(np.mean(y)),
        "metrics": all_metrics,
    }
    joblib.dump(artifact, output_dir / "unit_quality_model.joblib")

    summary = pd.DataFrame(all_metrics).T.sort_values("brier_score")
    summary.to_csv(output_dir / "model_comparison.csv")
    print(summary[["brier_score", "log_loss", "roc_auc", "average_precision"]])
    print(f"Saved model and evaluation files in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def raw_score_frame(artifact: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    """Return the boosted tree's native, uncalibrated good-unit score."""
    columns = list(artifact["feature_columns"])
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Extracted features are missing model inputs: {sorted(missing)}")
    x = frame[columns].replace([np.inf, -np.inf], np.nan)
    raw = artifact["model"].predict_proba(x)[:, 1]
    if not np.all(np.isfinite(raw)):
        raise ValueError("Model produced non-finite raw scores")
    if np.any((raw < 0.0) | (raw > 1.0)):
        raise ValueError("Model produced raw scores outside [0, 1]")
    return raw


def score_frame(artifact: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    """Return calibrated good-label probabilities for an extracted feature table."""
    raw = raw_score_frame(artifact, frame)
    probability = artifact["calibrator"].predict(raw)
    if not np.all(np.isfinite(probability)):
        raise ValueError("Model produced non-finite probabilities")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("Model produced probabilities outside [0, 1]")
    return probability


def write_phy_probability(
    path: Path, values: pd.DataFrame, *, overwrite: bool
) -> Path:
    """Atomically write a Phy cluster metadata TSV without touching labels."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace existing {path}; rerun with --overwrite if intended"
        )
    temporary = path.with_name(path.name + ".tmp")
    values.to_csv(temporary, sep="\t", index=False, float_format="%.8f")
    os.replace(temporary, path)
    return path

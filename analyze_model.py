#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


FEATURE_GROUPS = {
    "firing_rate_level_and_stability": [
        "mean_firing_rate_hz",
        "log_mean_firing_rate_hz",
        "phy_firing_rate_hz",
        "firing_rate_bin_cv",
        "log_firing_rate_adjacent_change_mean",
        "log_firing_rate_adjacent_change_max",
        "log_firing_rate_roughness",
        "low_rate_bin_fraction",
    ],
    "amplitude_level": [
        "mean_amplitude",
        "median_amplitude",
        "log_mean_amplitude",
        "phy_mean_amplitude",
        "amplitude_scale_mean",
        "amplitude_scale_median",
    ],
    "amplitude_distribution": [
        "amplitude_overall_cv",
        "amplitude_overall_skew",
        "amplitude_overall_excess_kurtosis",
        "amplitude_overall_qq_r2",
        "amplitude_lower_tail_deficit",
        "amplitude_bin_qq_r2_median",
        "amplitude_bin_qq_r2_p10",
        "amplitude_bin_abs_skew_median",
        "amplitude_bin_abs_skew_p90",
        "amplitude_bin_abs_excess_kurtosis_median",
        "amplitude_bin_abs_excess_kurtosis_p90",
        "amplitude_gaussian_valid_bin_fraction",
    ],
    "amplitude_stability": [
        "amplitude_bin_cv",
        "amplitude_adjacent_change_median",
        "amplitude_adjacent_change_p95",
        "amplitude_roughness",
    ],
    "presence": [
        "presence_ratio",
        "silent_bin_fraction",
        "max_silent_gap_fraction",
    ],
    "refractory_acg": [
        "isi_violation_fraction_1ms",
        "isi_violation_fraction_2ms",
        "acg_rate_0_1ms",
        "acg_rate_1_2ms",
        "acg_rate_2_5ms",
        "acg_rate_5_20ms",
        "acg_baseline_pair_count",
        "acg_trough_ratio_0_1_to_5_20",
        "acg_trough_ratio_0_2_to_5_20",
    ],
    "kilosort_and_basic": [
        "n_spikes",
        "log_n_spikes",
        "phy_contamination_pct",
        "phy_depth",
        "phy_peak_channel",
        "kilosort_label_good",
    ],
}


def _predict(artifact: dict[str, object], x: pd.DataFrame) -> np.ndarray:
    raw = artifact["model"].predict_proba(x)[:, 1]
    return artifact["calibrator"].predict(raw)


def _permuted(
    x: pd.DataFrame,
    columns: list[str],
    groups: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    result = x.copy()
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        shuffled = rng.permutation(indices)
        result.iloc[indices, result.columns.get_indexer(columns)] = (
            x.iloc[shuffled][columns].to_numpy()
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute descriptive within-session permutation importance."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=20260727)
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    frame = pd.read_csv(args.features)
    columns = list(artifact["feature_columns"])
    x = frame[columns].replace([np.inf, -np.inf], np.nan)
    y = frame["y"].astype(int).to_numpy()
    sessions = frame["session"].astype(str).to_numpy()
    baseline = brier_score_loss(y, _predict(artifact, x))
    rng = np.random.default_rng(args.random_state)

    def importance_rows(items: list[tuple[str, list[str]]]) -> pd.DataFrame:
        rows = []
        for name, requested_columns in items:
            selected = [column for column in requested_columns if column in columns]
            if not selected:
                continue
            deltas = []
            for _ in range(args.repeats):
                permuted = _permuted(x, selected, sessions, rng)
                score = brier_score_loss(y, _predict(artifact, permuted))
                deltas.append(score - baseline)
            rows.append(
                {
                    "name": name,
                    "features": ";".join(selected),
                    "brier_increase_mean": float(np.mean(deltas)),
                    "brier_increase_sd": float(np.std(deltas)),
                }
            )
        return pd.DataFrame(rows).sort_values("brier_increase_mean", ascending=False)

    feature_importance = importance_rows([(column, [column]) for column in columns])
    group_importance = importance_rows(list(FEATURE_GROUPS.items()))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_importance.to_csv(output_dir / "permutation_importance_features.csv", index=False)
    group_importance.to_csv(output_dir / "permutation_importance_groups.csv", index=False)
    print("Feature groups (descriptive training-set permutation importance):")
    print(group_importance[["name", "brier_increase_mean", "brier_increase_sd"]].to_string(index=False))
    print("\nTop individual features:")
    print(feature_importance[["name", "brier_increase_mean", "brier_increase_sd"]].head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


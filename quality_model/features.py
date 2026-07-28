from __future__ import annotations

import math
import re
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12
NORMAL = NormalDist()
QQ_PROBABILITIES = np.linspace(0.01, 0.99, 99)
NORMAL_QUANTILES = np.array(
    [NORMAL.inv_cdf(float(probability)) for probability in QQ_PROBABILITIES]
)
DAT_PATH_RE = re.compile(r"^\s*dat_path\s*=\s*[rRuUbBfF]*['\"](.*?)['\"]", re.MULTILINE)
SCALAR_RE = re.compile(r"^\s*(?P<name>\w+)\s*=\s*(?P<value>[^#\r\n]+)", re.MULTILINE)


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _safe_mean(values: np.ndarray) -> float:
    values = _finite(values)
    return float(np.mean(values)) if values.size else math.nan


def _safe_median(values: np.ndarray) -> float:
    values = _finite(values)
    return float(np.median(values)) if values.size else math.nan


def _safe_std(values: np.ndarray) -> float:
    values = _finite(values)
    return float(np.std(values)) if values.size else math.nan


def _safe_quantile(values: np.ndarray, q: float) -> float:
    values = _finite(values)
    return float(np.quantile(values, q)) if values.size else math.nan


def _skew_and_excess_kurtosis(values: np.ndarray) -> tuple[float, float]:
    values = _finite(values)
    if values.size < 4:
        return math.nan, math.nan
    centered = values - np.mean(values)
    variance = np.mean(centered**2)
    if variance <= EPS:
        return 0.0, -3.0
    skew = np.mean(centered**3) / variance**1.5
    kurtosis = np.mean(centered**4) / variance**2 - 3.0
    return float(skew), float(kurtosis)


def _normal_qq_r2(values: np.ndarray) -> float:
    """Squared correlation with normal quantiles; 1 is most Gaussian-like."""
    values = _finite(values)
    if values.size < 8 or np.std(values) <= EPS:
        return math.nan
    empirical_quantiles = np.quantile(values, QQ_PROBABILITIES)
    corr = np.corrcoef(empirical_quantiles, NORMAL_QUANTILES)[0, 1]
    return float(corr * corr) if np.isfinite(corr) else math.nan


def _lower_tail_deficit(values: np.ndarray, bins: int = 50) -> float:
    """Histogram asymmetry consistent with truncation of low amplitudes."""
    values = _finite(values)
    if values.size < 50 or np.ptp(values) <= EPS:
        return math.nan
    lo, hi = np.quantile(values, [0.005, 0.995])
    if hi <= lo:
        return math.nan
    counts, _ = np.histogram(values, bins=bins, range=(lo, hi))
    mode = int(np.argmax(counts))
    width = min(mode, bins - mode - 1)
    if width < 2:
        return 1.0
    left = counts[mode - width : mode][::-1].astype(float)
    right = counts[mode + 1 : mode + 1 + width].astype(float)
    missing = np.maximum(right - left, 0.0).sum()
    return float(np.clip(missing / max(values.size, 1), 0.0, 1.0))


def _parse_params(params_path: Path) -> dict[str, object]:
    text = params_path.read_text(encoding="utf-8", errors="replace")
    scalars: dict[str, object] = {}
    for match in SCALAR_RE.finditer(text):
        name = match.group("name")
        raw = match.group("value").strip().rstrip(".")
        if name in {"sample_rate", "n_channels_dat", "offset"}:
            try:
                scalars[name] = float(raw) if name == "sample_rate" else int(raw)
            except ValueError:
                pass
        elif name == "dtype":
            scalars[name] = raw.strip("'\"")
    dat_match = DAT_PATH_RE.search(text)
    if dat_match:
        raw_path = Path(dat_match.group(1))
        if not raw_path.is_absolute():
            raw_path = (params_path.parent / raw_path).resolve()
        scalars["dat_path"] = raw_path
    return scalars


def _find_phy_dir(root: Path, session: str, stream: str) -> Path:
    probe_dir = root / session / f"{session}_{stream}"
    matches = sorted(probe_dir.glob(f"phy_*_{stream}_ap"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one Phy folder under {probe_dir}, found {len(matches)}"
        )
    return matches[0]


def _load_cluster_info(phy_dir: Path) -> pd.DataFrame:
    path = phy_dir / "cluster_info.tsv"
    if path.exists():
        info = pd.read_csv(path, sep="\t")
    else:
        # Phy creates cluster_info.tsv after a session is opened/saved. Fresh
        # Kilosort folders still provide the core summaries as separate files.
        summary_files = {
            "Amplitude": "cluster_Amplitude.tsv",
            "ContamPct": "cluster_ContamPct.tsv",
            "KSLabel": "cluster_KSLabel.tsv",
        }
        frames = []
        for column, filename in summary_files.items():
            summary_path = phy_dir / filename
            if summary_path.exists():
                summary = pd.read_csv(summary_path, sep="\t")
                if "cluster_id" in summary.columns and column in summary.columns:
                    frames.append(summary[["cluster_id", column]])
        if not frames:
            return pd.DataFrame()
        info = frames[0]
        for frame in frames[1:]:
            info = info.merge(frame, on="cluster_id", how="outer")
    if "cluster_id" in info.columns:
        info = info.set_index("cluster_id", drop=False)
    return info


def _cluster_scalar(info: pd.DataFrame, cluster_id: int, column: str) -> float:
    if info.empty or column not in info.columns or cluster_id not in info.index:
        return math.nan
    value = info.loc[cluster_id, column]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    try:
        numeric = float(value)
        return numeric if np.isfinite(numeric) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _cluster_text(info: pd.DataFrame, cluster_id: int, column: str) -> str:
    if info.empty or column not in info.columns or cluster_id not in info.index:
        return ""
    value = info.loc[cluster_id, column]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    return str(value)


def _pair_count_between(times_s: np.ndarray, lower_s: float, upper_s: float) -> int:
    if times_s.size < 2:
        return 0
    lower = np.searchsorted(times_s, times_s + lower_s, side="right")
    upper = np.searchsorted(times_s, times_s + upper_s, side="right")
    return int(np.maximum(upper - lower, 0).sum())


def _refractory_features(times_s: np.ndarray) -> dict[str, float]:
    if times_s.size < 2:
        return {
            "isi_violation_fraction_1ms": math.nan,
            "isi_violation_fraction_2ms": math.nan,
            "acg_rate_0_1ms": math.nan,
            "acg_rate_1_2ms": math.nan,
            "acg_rate_2_5ms": math.nan,
            "acg_rate_5_20ms": math.nan,
            "acg_baseline_pair_count": 0.0,
            "acg_trough_ratio_0_1_to_5_20": math.nan,
            "acg_trough_ratio_0_2_to_5_20": math.nan,
        }
    isi = np.diff(times_s)
    counts = {
        "0_1": _pair_count_between(times_s, 0.0, 0.001),
        "1_2": _pair_count_between(times_s, 0.001, 0.002),
        "2_5": _pair_count_between(times_s, 0.002, 0.005),
        "5_20": _pair_count_between(times_s, 0.005, 0.020),
    }
    widths_ms = {"0_1": 1.0, "1_2": 1.0, "2_5": 3.0, "5_20": 15.0}
    # Conditional pair rates per spike per millisecond. This avoids making the
    # ACG features proxies for spike count or recording duration.
    rates = {
        key: counts[key] / (times_s.size * widths_ms[key]) for key in counts
    }
    baseline = rates["5_20"]
    enough_baseline_pairs = counts["5_20"] >= 10
    return {
        "isi_violation_fraction_1ms": float(np.mean(isi < 0.001)),
        "isi_violation_fraction_2ms": float(np.mean(isi < 0.002)),
        "acg_rate_0_1ms": float(rates["0_1"]),
        "acg_rate_1_2ms": float(rates["1_2"]),
        "acg_rate_2_5ms": float(rates["2_5"]),
        "acg_rate_5_20ms": float(baseline),
        "acg_baseline_pair_count": float(counts["5_20"]),
        "acg_trough_ratio_0_1_to_5_20": float(rates["0_1"] / baseline)
        if enough_baseline_pairs and baseline > 0
        else math.nan,
        "acg_trough_ratio_0_2_to_5_20": float(
            ((rates["0_1"] + rates["1_2"]) / 2.0) / baseline
        )
        if enough_baseline_pairs and baseline > 0
        else math.nan,
    }


def _temporal_features(
    times_s: np.ndarray,
    amplitudes: np.ndarray,
    duration_s: float,
    bin_seconds: float,
    min_gaussian_spikes: int,
) -> dict[str, float]:
    n_bins = max(10, int(math.ceil(duration_s / bin_seconds)))
    n_bins = min(n_bins, 200)
    edges = np.linspace(0.0, duration_s, n_bins + 1)
    bin_duration = duration_s / n_bins
    # Unit spike times are already sorted. Search once for bin boundaries and
    # slice directly, rather than rescanning every spike for every time bin.
    boundaries = np.searchsorted(times_s, edges, side="left")
    boundaries[-1] = times_s.size
    counts = np.diff(boundaries).astype(float)
    rates = counts / max(bin_duration, EPS)
    active = counts > 0

    medians = np.full(n_bins, np.nan)
    qq_r2 = np.full(n_bins, np.nan)
    abs_skew = np.full(n_bins, np.nan)
    abs_kurt = np.full(n_bins, np.nan)
    for bin_id in np.flatnonzero(active):
        values = amplitudes[boundaries[bin_id] : boundaries[bin_id + 1]]
        medians[bin_id] = np.median(values)
        if values.size >= min_gaussian_spikes:
            skew, kurtosis = _skew_and_excess_kurtosis(values)
            qq_r2[bin_id] = _normal_qq_r2(values)
            abs_skew[bin_id] = abs(skew)
            abs_kurt[bin_id] = abs(kurtosis)

    scale = abs(_safe_median(medians)) + EPS
    adjacent_mask = np.isfinite(medians[:-1]) & np.isfinite(medians[1:])
    adjacent_changes = (
        np.abs(np.diff(medians)[adjacent_mask]) / scale
        if adjacent_mask.any()
        else np.array([], dtype=float)
    )
    triple_mask = np.isfinite(medians[:-2]) & np.isfinite(medians[1:-1]) & np.isfinite(medians[2:])
    second_diffs = (
        np.diff(medians, n=2)[triple_mask] / scale
        if triple_mask.any()
        else np.array([], dtype=float)
    )

    log_rates = np.log1p(rates)
    positive_rates = rates[rates > 0]
    rate_scale = _safe_median(positive_rates) + EPS
    quiet_threshold = 0.1 * rate_scale

    max_silent_run = 0
    current_run = 0
    for is_active in active:
        if is_active:
            current_run = 0
        else:
            current_run += 1
            max_silent_run = max(max_silent_run, current_run)

    return {
        "time_bin_count": float(n_bins),
        "presence_ratio": float(np.mean(active)),
        "silent_bin_fraction": float(np.mean(~active)),
        "max_silent_gap_fraction": float(max_silent_run / n_bins),
        "low_rate_bin_fraction": float(np.mean(rates < quiet_threshold)),
        "firing_rate_bin_cv": float(np.std(rates) / (np.mean(rates) + EPS)),
        "log_firing_rate_adjacent_change_mean": _safe_mean(np.abs(np.diff(log_rates))),
        "log_firing_rate_adjacent_change_max": _safe_quantile(np.abs(np.diff(log_rates)), 0.95),
        "log_firing_rate_roughness": float(np.sqrt(np.mean(np.diff(log_rates, n=2) ** 2)))
        if n_bins >= 3
        else math.nan,
        "amplitude_bin_cv": float(_safe_std(medians) / scale),
        "amplitude_adjacent_change_median": _safe_median(adjacent_changes),
        "amplitude_adjacent_change_p95": _safe_quantile(adjacent_changes, 0.95),
        "amplitude_roughness": float(np.sqrt(np.mean(second_diffs**2)))
        if second_diffs.size
        else math.nan,
        "amplitude_bin_qq_r2_median": _safe_median(qq_r2),
        "amplitude_bin_qq_r2_p10": _safe_quantile(qq_r2, 0.10),
        "amplitude_bin_abs_skew_median": _safe_median(abs_skew),
        "amplitude_bin_abs_skew_p90": _safe_quantile(abs_skew, 0.90),
        "amplitude_bin_abs_excess_kurtosis_median": _safe_median(abs_kurt),
        "amplitude_bin_abs_excess_kurtosis_p90": _safe_quantile(abs_kurt, 0.90),
        "amplitude_gaussian_valid_bin_fraction": float(np.mean(np.isfinite(qq_r2))),
    }


def _infer_peak_channel(
    cluster_id: int,
    spike_template_ids: np.ndarray,
    templates: np.ndarray,
    cluster_info: pd.DataFrame,
    n_channels: int,
) -> int | None:
    from_info = _cluster_scalar(cluster_info, cluster_id, "ch")
    if np.isfinite(from_info) and 0 <= int(from_info) < n_channels:
        return int(from_info)
    if spike_template_ids.size == 0:
        return None
    template_id = int(np.bincount(spike_template_ids.astype(int)).argmax())
    if not 0 <= template_id < templates.shape[0]:
        return None
    template = np.asarray(templates[template_id])
    ptp = np.ptp(template, axis=0)
    return int(np.argmax(ptp)) if ptp.size else None


def _waveform_features(
    spike_samples: np.ndarray,
    peak_channel: int | None,
    raw: np.memmap | None,
    n_channels: int,
    waveform_samples: int,
    time_groups: int,
    waveforms_per_group: int,
) -> dict[str, float]:
    empty = {
        "waveform_peak_to_peak_raw": math.nan,
        "waveform_time_correlation_median": math.nan,
        "waveform_time_correlation_min": math.nan,
        "waveform_peak_to_peak_cv": math.nan,
        "waveform_residual_noise_ratio": math.nan,
        "waveform_individual_correlation_median": math.nan,
        "waveform_valid_time_group_fraction": 0.0,
    }
    if raw is None or peak_channel is None or spike_samples.size < 5:
        return empty
    half = waveform_samples // 2
    offsets = np.arange(-half, half + 1, dtype=np.int64)
    total_samples = raw.size // n_channels
    valid = spike_samples[(spike_samples >= half) & (spike_samples < total_samples - half - 1)]
    if valid.size < 5:
        return empty

    boundaries = np.linspace(valid.min(), valid.max() + 1, time_groups + 1)
    group_means: list[np.ndarray] = []
    group_ptp: list[float] = []
    individual_corrs: list[float] = []
    residual_ratios: list[float] = []
    raw_2d = raw.reshape(-1, n_channels)

    for group in range(time_groups):
        group_spikes = valid[(valid >= boundaries[group]) & (valid < boundaries[group + 1])]
        if group_spikes.size < 3:
            continue
        if group_spikes.size > waveforms_per_group:
            positions = np.linspace(0, group_spikes.size - 1, waveforms_per_group).astype(int)
            group_spikes = group_spikes[positions]
        sample_indices = group_spikes[:, None] + offsets[None, :]
        waveforms = np.asarray(raw_2d[sample_indices, peak_channel], dtype=np.float32)
        means = np.mean(waveforms, axis=0)
        baseline_n = max(2, waveform_samples // 6)
        means = means - np.mean(means[:baseline_n])
        waveforms = waveforms - np.mean(waveforms[:, :baseline_n], axis=1, keepdims=True)
        ptp = float(np.ptp(means))
        if ptp <= EPS:
            continue
        group_means.append(means)
        group_ptp.append(ptp)
        residual_ratios.append(float(np.mean(np.std(waveforms - means, axis=0)) / ptp))

        centered_waveforms = waveforms - waveforms.mean(axis=1, keepdims=True)
        centered_mean = means - means.mean()
        denominators = np.linalg.norm(centered_waveforms, axis=1) * np.linalg.norm(centered_mean)
        corr = (centered_waveforms @ centered_mean) / np.maximum(denominators, EPS)
        individual_corrs.extend(corr[np.isfinite(corr)].tolist())

    if not group_means:
        return empty
    global_mean = np.mean(np.stack(group_means), axis=0)
    centered_global = global_mean - global_mean.mean()
    correlations = []
    for means in group_means:
        centered = means - means.mean()
        denom = np.linalg.norm(centered) * np.linalg.norm(centered_global)
        if denom > EPS:
            correlations.append(float(np.dot(centered, centered_global) / denom))
    return {
        "waveform_peak_to_peak_raw": float(np.ptp(global_mean)),
        "waveform_time_correlation_median": _safe_median(np.asarray(correlations)),
        "waveform_time_correlation_min": _safe_quantile(np.asarray(correlations), 0.10),
        "waveform_peak_to_peak_cv": float(np.std(group_ptp) / (np.mean(group_ptp) + EPS)),
        "waveform_residual_noise_ratio": _safe_median(np.asarray(residual_ratios)),
        "waveform_individual_correlation_median": _safe_median(np.asarray(individual_corrs)),
        "waveform_valid_time_group_fraction": float(len(group_means) / time_groups),
    }


def extract_probe_features(
    root: Path,
    session: str,
    stream: str,
    region: str,
    *,
    bin_seconds: float = 60.0,
    min_gaussian_spikes: int = 20,
    waveform_samples: int = 61,
    waveform_time_groups: int = 10,
    waveforms_per_group: int = 40,
    include_waveforms: bool = True,
    max_clusters: int | None = None,
    all_clusters: bool = False,
) -> pd.DataFrame:
    phy_dir = _find_phy_dir(root, session, stream)
    cluster_info = _load_cluster_info(phy_dir)
    group_path = phy_dir / "cluster_group.tsv"
    saved_groups = (
        pd.read_csv(group_path, sep="\t")
        if group_path.exists()
        else pd.DataFrame(columns=["cluster_id", "group"])
    )
    if all_clusters:
        if cluster_info.empty or "cluster_id" not in cluster_info.columns:
            raise ValueError(f"cluster_info.tsv has no cluster_id column in {phy_dir}")
        labels = (
            cluster_info[["cluster_id"]]
            .reset_index(drop=True)
            .drop_duplicates()
            .copy()
        )
        if not saved_groups.empty:
            labels = labels.merge(
                saved_groups[["cluster_id", "group"]].drop_duplicates("cluster_id"),
                on="cluster_id",
                how="left",
            )
        else:
            labels["group"] = np.nan
        labels["group"] = labels["group"].fillna("unlabeled")
    else:
        labels = saved_groups[saved_groups["group"].isin(["good", "mua", "noise"])].copy()
    if max_clusters is not None:
        labels = labels.head(max_clusters)

    params = _parse_params(phy_dir / "params.py")
    sample_rate = float(params.get("sample_rate", 30_000.0))
    n_channels = int(params.get("n_channels_dat", 384))
    dtype = np.dtype(str(params.get("dtype", "int16")))
    dat_path = params.get("dat_path")
    raw: np.memmap | None = None
    duration_s = math.nan
    if dat_path and Path(dat_path).exists():
        raw_value_count = Path(dat_path).stat().st_size // dtype.itemsize
        duration_s = (raw_value_count // n_channels) / sample_rate
        if include_waveforms:
            raw = np.memmap(Path(dat_path), dtype=dtype, mode="r")

    spike_clusters = np.load(phy_dir / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    spike_times = np.load(phy_dir / "spike_times.npy", mmap_mode="r").reshape(-1)
    spike_amplitudes = np.load(phy_dir / "amplitudes.npy", mmap_mode="r").reshape(-1)
    spike_templates = (
        np.load(phy_dir / "spike_templates.npy", mmap_mode="r").reshape(-1)
        if include_waveforms
        else None
    )
    templates = (
        np.load(phy_dir / "templates.npy", mmap_mode="r")
        if include_waveforms
        else None
    )
    lengths = [spike_clusters.size, spike_times.size, spike_amplitudes.size]
    if spike_templates is not None:
        lengths.append(spike_templates.size)
    if len(set(lengths)) != 1:
        raise ValueError(f"Spike arrays have inconsistent lengths in {phy_dir}")
    if not np.isfinite(duration_s):
        duration_s = float(np.max(spike_times) / sample_rate) if spike_times.size else 0.0

    records: list[dict[str, object]] = []

    for label_row in labels.itertuples(index=False):
        cluster_id = int(label_row.cluster_id)
        # A boolean scan is faster and far less memory-intensive here than
        # sorting tens of millions of cluster assignments. These arrays are
        # memory-mapped and quickly enter the operating-system file cache.
        spike_indices = np.flatnonzero(spike_clusters == cluster_id)
        samples = np.asarray(spike_times[spike_indices], dtype=np.int64)
        sort_order = np.argsort(samples, kind="stable")
        samples = samples[sort_order]
        times_s = samples.astype(float) / sample_rate
        amplitude_scale = np.asarray(spike_amplitudes[spike_indices], dtype=float)[sort_order]
        template_ids = (
            np.asarray(spike_templates[spike_indices], dtype=int)[sort_order]
            if spike_templates is not None
            else np.array([], dtype=int)
        )

        phy_amplitude = _cluster_scalar(cluster_info, cluster_id, "Amplitude")
        mean_scale = _safe_mean(amplitude_scale)
        if np.isfinite(phy_amplitude) and np.isfinite(mean_scale) and abs(mean_scale) > EPS:
            amplitudes = amplitude_scale * phy_amplitude / mean_scale
        else:
            amplitudes = amplitude_scale

        mean_firing_rate = samples.size / max(duration_s, EPS)
        amplitude_skew, amplitude_kurtosis = _skew_and_excess_kurtosis(amplitudes)
        ks_label = _cluster_text(cluster_info, cluster_id, "KSLabel").lower()
        record: dict[str, object] = {
            "session": session,
            "stream": stream,
            "region": region,
            "phy_dir": str(phy_dir),
            "cluster_id": cluster_id,
            "label": str(label_row.group),
            "y": 1.0
            if label_row.group == "good"
            else (0.0 if label_row.group in {"mua", "noise"} else math.nan),
            "session_duration_s": float(duration_s),
            "n_spikes": int(samples.size),
            "log_n_spikes": float(np.log1p(samples.size)),
            "mean_firing_rate_hz": float(mean_firing_rate),
            "log_mean_firing_rate_hz": float(np.log1p(mean_firing_rate)),
            "mean_amplitude": _safe_mean(amplitudes),
            "median_amplitude": _safe_median(amplitudes),
            "log_mean_amplitude": float(np.log1p(max(_safe_mean(amplitudes), 0.0))),
            "amplitude_scale_mean": mean_scale,
            "amplitude_scale_median": _safe_median(amplitude_scale),
            "amplitude_overall_cv": float(_safe_std(amplitudes) / (abs(_safe_mean(amplitudes)) + EPS)),
            "amplitude_overall_skew": amplitude_skew,
            "amplitude_overall_excess_kurtosis": amplitude_kurtosis,
            "amplitude_overall_qq_r2": _normal_qq_r2(amplitudes),
            "amplitude_lower_tail_deficit": _lower_tail_deficit(amplitudes),
            "phy_mean_amplitude": phy_amplitude,
            "phy_contamination_pct": _cluster_scalar(cluster_info, cluster_id, "ContamPct"),
            "phy_depth": _cluster_scalar(cluster_info, cluster_id, "depth"),
            "phy_peak_channel": _cluster_scalar(cluster_info, cluster_id, "ch"),
            "phy_firing_rate_hz": _cluster_scalar(cluster_info, cluster_id, "fr"),
            "kilosort_label_good": float(ks_label == "good") if ks_label else math.nan,
        }
        record.update(_refractory_features(times_s))
        record.update(
            _temporal_features(
                times_s,
                amplitudes,
                duration_s,
                bin_seconds,
                min_gaussian_spikes,
            )
        )
        peak_channel = (
            _infer_peak_channel(
                cluster_id, template_ids, templates, cluster_info, n_channels
            )
            if include_waveforms and templates is not None
            else None
        )
        record.update(
            _waveform_features(
                samples,
                peak_channel,
                raw if include_waveforms else None,
                n_channels,
                waveform_samples,
                waveform_time_groups,
                waveforms_per_group,
            )
        )
        records.append(record)

    return pd.DataFrame.from_records(records)


def manifest_rows(
    manifest_path: Path, selection_column: str = "curated"
) -> Iterable[tuple[str, str, str]]:
    manifest = pd.read_csv(manifest_path, dtype=str)
    required = {"session", "stream", "region", selection_column}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    for row in manifest[manifest[selection_column].eq("1")].itertuples(index=False):
        yield str(row.session), str(row.stream), str(row.region)


def curated_manifest_rows(manifest_path: Path) -> Iterable[tuple[str, str, str]]:
    yield from manifest_rows(manifest_path, "curated")

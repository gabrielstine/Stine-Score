#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from quality_model.features import extract_probe_features, manifest_rows
from quality_model.features import _find_phy_dir as find_phy_dir
from quality_model.features import _load_cluster_info as load_cluster_info


def score_frame(artifact: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    columns = list(artifact["feature_columns"])
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Extracted features are missing model inputs: {sorted(missing)}")
    x = frame[columns].replace([np.inf, -np.inf], np.nan)
    raw = artifact["model"].predict_proba(x)[:, 1]
    probability = artifact["calibrator"].predict(raw)
    if not np.all(np.isfinite(probability)):
        raise ValueError("Model produced non-finite probabilities")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("Model produced probabilities outside [0, 1]")
    return probability


def write_phy_probability(
    path: Path, values: pd.DataFrame, *, overwrite: bool
) -> str:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace existing {path}; rerun with --overwrite if intended"
        )
    temporary = path.with_name(path.name + ".tmp")
    values.to_csv(temporary, sep="\t", index=False, float_format="%.8f")
    os.replace(temporary, path)
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score every sorted manifest probe and optionally add a Phy column."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/all_probe_features")
    )
    parser.add_argument(
        "--column-name", default="good_probability", help="Phy metadata column name"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write cluster_<column-name>.tsv into each Phy folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing probability TSV. Only used with --apply.",
    )
    parser.add_argument("--max-probes", type=int)
    parser.add_argument("--max-clusters", type=int)
    parser.add_argument("--only-probe", action="append", metavar="SESSION:STREAM")
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse complete local per-probe score CSV files when available.",
    )
    args = parser.parse_args()

    if not args.column_name.replace("_", "").isalnum():
        raise ValueError("Column name may contain only letters, numbers, and underscores")

    root = args.root.resolve()
    manifest = args.manifest or root / "probe_manifest.csv"
    rows = list(manifest_rows(manifest, "sorted"))
    if args.only_probe:
        requested = {tuple(value.rsplit(":", 1)) for value in args.only_probe}
        rows = [row for row in rows if (row[0], row[1]) in requested]
    if args.max_probes is not None:
        rows = rows[: args.max_probes]
    if not rows:
        raise ValueError("No sorted probes matched the request")

    target_paths = [
        find_phy_dir(root, session, stream)
        / f"cluster_{args.column_name}.tsv"
        for session, stream, _ in rows
    ]
    existing_targets = [path for path in target_paths if path.exists()]
    if args.apply and existing_targets and not args.overwrite:
        preview = "\n".join(str(path) for path in existing_targets[:10])
        raise FileExistsError(
            f"{len(existing_targets)} target files already exist. No files were written. "
            f"Use --overwrite if replacement is intended.\n{preview}"
        )

    artifact = joblib.load(args.model)
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[tuple[Path, pd.DataFrame]] = []
    combined: list[pd.DataFrame] = []

    # Complete and validate all extraction/scoring before writing to U:.
    for index, (session, stream, region) in enumerate(rows, start=1):
        cache_path = cache_dir / f"{session}__{stream}.csv"
        if args.reuse_cache and cache_path.exists() and args.max_clusters is None:
            print(f"[{index}/{len(rows)}] cached {session} {stream} ({region})", flush=True)
            scored = pd.read_csv(cache_path)
            required_cache = {
                "session",
                "stream",
                "region",
                "cluster_id",
                "label",
                args.column_name,
            }
            missing_cache = required_cache - set(scored.columns)
            if missing_cache:
                raise ValueError(
                    f"Cache {cache_path} is missing columns: {sorted(missing_cache)}"
                )
        else:
            print(f"[{index}/{len(rows)}] scoring {session} {stream} ({region})", flush=True)
            frame = extract_probe_features(
                root,
                session,
                stream,
                region,
                include_waveforms=False,
                max_clusters=args.max_clusters,
                all_clusters=True,
            )
            probability = score_frame(artifact, frame)
            scored = frame[["session", "stream", "region", "cluster_id", "label"]].copy()
            scored[args.column_name] = probability
            scored.to_csv(cache_path, index=False)
        if scored["cluster_id"].duplicated().any():
            raise ValueError(f"Duplicate cluster IDs in {session} {stream}")
        probability_values = scored[args.column_name].to_numpy(dtype=float)
        if not np.all(np.isfinite(probability_values)) or np.any(
            (probability_values < 0.0) | (probability_values > 1.0)
        ):
            raise ValueError(f"Invalid cached probabilities in {session} {stream}")
        phy_dir = find_phy_dir(root, session, stream)
        expected_info = load_cluster_info(phy_dir)
        expected_ids = set(expected_info["cluster_id"].astype(int).tolist())
        scored_ids = set(scored["cluster_id"].astype(int).tolist())
        if scored_ids != expected_ids:
            raise ValueError(
                f"Cache/feature cluster IDs do not match the Phy summaries in "
                f"{session} {stream}: scored={len(scored_ids)}, expected={len(expected_ids)}"
            )
        phy_values = scored[["cluster_id", args.column_name]].sort_values("cluster_id")
        prepared.append((phy_dir / f"cluster_{args.column_name}.tsv", phy_values))
        combined.append(scored)

    combined_frame = pd.concat(combined, ignore_index=True)
    combined_path = cache_dir.parent / "all_probe_probabilities.csv"
    combined_frame.to_csv(combined_path, index=False)
    print(
        f"Prepared {len(combined_frame)} cluster probabilities across {len(rows)} probes."
    )
    print(f"Local combined output: {combined_path}")

    if not args.apply:
        print("Dry run complete; no files were written to the Phy folders.")
        return 0

    for path, values in prepared:
        write_phy_probability(path, values, overwrite=args.overwrite)
        print(f"Wrote {len(values)} rows: {path}", flush=True)
    print(f"Added cluster_{args.column_name}.tsv to {len(prepared)} Phy folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

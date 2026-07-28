#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from quality_model.features import (
    extract_phy_features,
    iter_phy_dirs,
    phy_cache_stem,
    phy_folder_metadata,
)
from quality_model.features import _load_cluster_info as load_cluster_info
from quality_model.scoring import raw_score_frame, score_frame, write_phy_probability


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score every Phy output folder found under a base directory."
    )
    parser.add_argument("root", type=Path, help="Base directory containing Phy folders")
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/discovered_probe_scores")
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write calibrated and uncalibrated score TSVs into each Phy folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing probability TSV. Only used with --apply.",
    )
    parser.add_argument("--max-probes", type=int)
    parser.add_argument(
        "--start-probe",
        type=int,
        default=0,
        help="Zero-based index of the first discovered Phy folder to score.",
    )
    parser.add_argument("--max-clusters", type=int)
    parser.add_argument("--only-phy-dir", action="append", metavar="PATH")
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse complete local per-probe score CSV files when available.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    phy_dirs = [Path(path).resolve() for path in args.only_phy_dir] if args.only_phy_dir else iter_phy_dirs(root)
    if args.start_probe < 0:
        raise ValueError("--start-probe must be zero or greater")
    phy_dirs = phy_dirs[args.start_probe :]
    if args.max_probes is not None:
        phy_dirs = phy_dirs[: args.max_probes]
    if not phy_dirs:
        raise ValueError(f"No valid Phy output folders found under {root}")

    target_paths = [
        output_path
        for phy_dir in phy_dirs
        for output_path in (
            phy_dir / "cluster_good_probability.tsv",
            phy_dir / "cluster_good_unit_score.tsv",
        )
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
    for index, phy_dir in enumerate(phy_dirs, start=1):
        session, stream, region = phy_folder_metadata(phy_dir)
        cache_path = cache_dir / f"{phy_cache_stem(root, phy_dir)}.csv"
        if args.reuse_cache and cache_path.exists() and args.max_clusters is None:
            print(f"[{index}/{len(phy_dirs)}] cached {phy_dir}", flush=True)
            scored = pd.read_csv(cache_path)
            required_cache = {
                "session",
                "stream",
                "region",
                "cluster_id",
                "label",
                "good_probability",
                "good_unit_score",
            }
            missing_cache = required_cache - set(scored.columns)
            if missing_cache:
                raise ValueError(
                    f"Cache {cache_path} is missing columns: {sorted(missing_cache)}"
                )
        else:
            print(f"[{index}/{len(phy_dirs)}] scoring {phy_dir}", flush=True)
            frame = extract_phy_features(
                phy_dir,
                session=session,
                stream=stream,
                region=region,
                include_waveforms=False,
                max_clusters=args.max_clusters,
                all_clusters=True,
            )
            probability = score_frame(artifact, frame)
            raw_score = raw_score_frame(artifact, frame)
            scored = frame[["session", "stream", "region", "cluster_id", "label"]].copy()
            scored["good_probability"] = probability
            scored["good_unit_score"] = raw_score
            scored.to_csv(cache_path, index=False)
        if scored["cluster_id"].duplicated().any():
            raise ValueError(f"Duplicate cluster IDs in {session} {stream}")
        for column_name in ("good_probability", "good_unit_score"):
            values = scored[column_name].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
                raise ValueError(f"Invalid cached {column_name} values in {session} {stream}")
        expected_info = load_cluster_info(phy_dir)
        expected_ids = set(expected_info["cluster_id"].astype(int).tolist())
        scored_ids = set(scored["cluster_id"].astype(int).tolist())
        if scored_ids != expected_ids:
            raise ValueError(
                f"Cache/feature cluster IDs do not match the Phy summaries in "
                f"{phy_dir}: scored={len(scored_ids)}, expected={len(expected_ids)}"
            )
        for column_name in ("good_probability", "good_unit_score"):
            phy_values = scored[["cluster_id", column_name]].sort_values("cluster_id")
            prepared.append((phy_dir / f"cluster_{column_name}.tsv", phy_values))
        combined.append(scored)

    combined_frame = pd.concat(combined, ignore_index=True)
    combined_path = cache_dir / "all_probe_probabilities.csv"
    combined_frame.to_csv(combined_path, index=False)
    print(
        f"Prepared calibrated probabilities and raw scores for {len(combined_frame)} clusters across {len(phy_dirs)} probes."
    )
    print(f"Local combined output: {combined_path}")

    if not args.apply:
        print("Dry run complete; no files were written to the Phy folders.")
        return 0

    for path, values in prepared:
        write_phy_probability(path, values, overwrite=args.overwrite)
        print(f"Wrote {len(values)} rows: {path}", flush=True)
    print(f"Added both score TSVs to {len(phy_dirs)} Phy folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

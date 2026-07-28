#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quality_model.features import (
    extract_phy_features,
    iter_phy_dirs,
    phy_cache_stem,
    phy_folder_metadata,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract training features from labeled Phy/Kilosort folders."
    )
    parser.add_argument("root", type=Path, help="Base directory containing Phy folders")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/discovered_features")
    )
    parser.add_argument("--bin-seconds", type=float, default=60.0)
    parser.add_argument("--max-probes", type=int)
    parser.add_argument("--max-clusters", type=int)
    parser.add_argument(
        "--only-phy-dir",
        action="append",
        metavar="PATH",
        help="Extract only a specific Phy output folder; may be repeated.",
    )
    parser.add_argument(
        "--include-waveforms",
        action="store_true",
        help="Sample raw waveforms over time. Disabled by default because it is slow.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute cached per-probe CSV files."
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    phy_dirs = [Path(path).resolve() for path in args.only_phy_dir] if args.only_phy_dir else iter_phy_dirs(root)
    if args.max_probes is not None:
        phy_dirs = phy_dirs[: args.max_probes]
    if not phy_dirs:
        raise RuntimeError(f"No valid Phy output folders found under {root}")
    probe_files: list[Path] = []

    for index, phy_dir in enumerate(phy_dirs, start=1):
        session, stream, region = phy_folder_metadata(phy_dir)
        output = output_dir / f"{phy_cache_stem(root, phy_dir)}.csv"
        probe_files.append(output)
        if output.exists() and not args.overwrite:
            print(f"[{index}/{len(phy_dirs)}] cached {phy_dir}", flush=True)
            continue
        print(f"[{index}/{len(phy_dirs)}] extracting {phy_dir}", flush=True)
        frame = extract_phy_features(
            phy_dir,
            session=session,
            stream=stream,
            region=region,
            bin_seconds=args.bin_seconds,
            include_waveforms=args.include_waveforms,
            max_clusters=args.max_clusters,
        )
        if frame.empty:
            print("  skipped: no saved good/mua/noise labels", flush=True)
            continue
        frame.to_csv(output, index=False)
        print(f"  wrote {len(frame)} units to {output}", flush=True)

    # Combine every cached probe, including caches from earlier resumable runs.
    frames = [pd.read_csv(path) for path in probe_files if path.exists()]
    if not frames:
        raise RuntimeError("No feature files were produced")
    combined = pd.concat(frames, ignore_index=True)
    combined_path = output_dir / "features.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Combined {len(combined)} units in {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

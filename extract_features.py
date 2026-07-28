#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quality_model.features import curated_manifest_rows, extract_probe_features


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract unit-quality features from curated Phy/Kilosort folders."
    )
    parser.add_argument("root", type=Path, help=r"Data root, for example U:\Brass")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest path; defaults to <root>/probe_manifest.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/features")
    )
    parser.add_argument("--bin-seconds", type=float, default=60.0)
    parser.add_argument("--max-probes", type=int)
    parser.add_argument("--max-clusters", type=int)
    parser.add_argument(
        "--only-probe",
        action="append",
        metavar="SESSION:STREAM",
        help="Extract only a specific probe; may be repeated.",
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
    manifest = args.manifest or root / "probe_manifest.csv"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(curated_manifest_rows(manifest))
    if args.only_probe:
        requested = {tuple(value.rsplit(":", 1)) for value in args.only_probe}
        rows = [row for row in rows if (row[0], row[1]) in requested]
    if args.max_probes is not None:
        rows = rows[: args.max_probes]
    probe_files: list[Path] = []

    for index, (session, stream, region) in enumerate(rows, start=1):
        output = output_dir / f"{session}__{stream}.csv"
        probe_files.append(output)
        if output.exists() and not args.overwrite:
            print(f"[{index}/{len(rows)}] cached {session} {stream}", flush=True)
            continue
        print(f"[{index}/{len(rows)}] extracting {session} {stream} ({region})", flush=True)
        frame = extract_probe_features(
            root,
            session,
            stream,
            region,
            bin_seconds=args.bin_seconds,
            include_waveforms=args.include_waveforms,
            max_clusters=args.max_clusters,
        )
        frame.to_csv(output, index=False)
        print(f"  wrote {len(frame)} units to {output}", flush=True)

    # Combine every cached probe, including caches from earlier resumable runs.
    frames = [pd.read_csv(path) for path in sorted(output_dir.glob("*.csv"))]
    if not frames:
        raise RuntimeError("No feature files were produced")
    combined = pd.concat(frames, ignore_index=True)
    combined_path = output_dir.parent / "features.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Combined {len(combined)} units in {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

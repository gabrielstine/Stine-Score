#!/usr/bin/env python
"""
Fix Phy params.py raw-data paths under a session/data root.

The bug this repairs is usually caused by Windows paths being written into
params.py with single backslashes, e.g. U:\Brass\20260706...\traces...
Python then treats pieces like \202 and \t as escape sequences, so Phy looks
for a mangled path.

By default this writes a portable relative path from each phy output folder:
../preprocess_motion_corrected/traces_cached_seg0.raw
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_DAT_PATH = "../preprocess_motion_corrected/traces_cached_seg0.raw"
DAT_PATH_RE = re.compile(
    r"^(?P<prefix>\s*dat_path\s*=\s*)(?P<string_prefix>[rRuUbBfF]*)(?P<quote>['\"])(?P<value>.*?)(?P=quote)(?P<suffix>.*)$",
    re.MULTILINE,
)


def fix_params_file(params_file: Path, apply: bool) -> tuple[bool, str]:
    text = params_file.read_text(encoding="utf-8", errors="replace")
    match = DAT_PATH_RE.search(text)
    if not match:
        return False, "no dat_path assignment found"

    phy_folder = params_file.parent
    expected_raw = (phy_folder / DEFAULT_DAT_PATH).resolve()
    if not expected_raw.exists():
        return False, f"raw file not found at expected relative path: {expected_raw}"

    old_value = match.group("value")
    if old_value == DEFAULT_DAT_PATH:
        return False, "already correct"

    new_line = (
        f"{match.group('prefix')}'{DEFAULT_DAT_PATH}'{match.group('suffix')}"
    )
    new_text = text[: match.start()] + new_line + text[match.end() :]

    if apply:
        params_file.write_text(new_text, encoding="utf-8", newline="")

    return True, f"{old_value!r} -> {DEFAULT_DAT_PATH!r}"


def iter_params_files(root: Path):
    yield from root.rglob("params.py")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fix dat_path lines in Phy params.py files."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root to scan, e.g. U:\\Brass or a single session folder.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually modify files. Without this, only prints a dry run.",
    )
    parser.add_argument(
        "--all-params",
        action="store_true",
        help="Also inspect params.py files outside phy_KS2* folders.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        print(f"Root does not exist: {root}")
        return 2

    checked = changed = skipped = 0
    for params_file in iter_params_files(root):
        if not args.all_params and "phy_KS2" not in str(params_file.parent):
            continue

        checked += 1
        did_change, message = fix_params_file(params_file, apply=args.apply)
        if did_change:
            changed += 1
            verb = "Fixed" if args.apply else "Would fix"
            print(f"{verb}: {params_file}")
            print(f"  {message}")
        else:
            skipped += 1
            print(f"Skipped: {params_file}")
            print(f"  {message}")

    mode = "Applied" if args.apply else "Dry run"
    print(
        f"\n{mode} complete. Checked {checked}, "
        f"{'changed' if args.apply else 'would change'} {changed}, skipped {skipped}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

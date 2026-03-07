#!/usr/bin/env python3
"""Install dependencies for RT-Gesture without using conda install."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


DEFAULT_CONDA_PREFIX = "/mnt/ext_drive/workspace/env/conda/torch2.4.1"
BASE_PACKAGES = [
    "PyQt6>=6.5,<7",
    "pyqtgraph>=0.13,<0.14",
    "msgpack>=1.0,<2",
]


def build_command(prefix: str | None, packages: list[str]) -> list[str]:
    if prefix:
        return ["conda", "run", "-p", prefix, "pip", "install", *packages]
    return [sys.executable, "-m", "pip", "install", *packages]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        default=DEFAULT_CONDA_PREFIX,
        help="Conda prefix used by `conda run -p`. Use empty value to install with current Python.",
    )
    parser.add_argument(
        "--no-conda-run",
        action="store_true",
        help="Install into the current Python environment directly.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command only, do not execute.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prefix = None if args.no_conda_run else (args.prefix or None)
    command = build_command(prefix=prefix, packages=BASE_PACKAGES)
    print("Installing dependencies with command:")
    print("  " + shlex.join(command))
    if args.dry_run:
        return 0
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


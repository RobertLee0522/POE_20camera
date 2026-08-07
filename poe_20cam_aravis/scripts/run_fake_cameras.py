#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_fake_cameras.py
===========================
Spawn N Aravis fake GigE cameras so the 20-camera UI can be exercised
without hardware.

    python3 scripts/run_fake_cameras.py 20        # then, in another shell:
    ./run.sh

Each camera is a separate ``arv-fake-gv-camera-0.10`` process bound to the
loopback interface with its own serial number.  Ctrl-C stops them all.

The single in-process fake camera (``./run.sh --fake``) is enough to smoke-test
the pipeline; this script is what you want when you need a realistic 20-tile
layout, batching behaviour and per-camera logging.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
from typing import List

_BINARY_CANDIDATES = [
    "arv-fake-gv-camera-0.10",
    "arv-fake-gv-camera-0.8",
    "arv-fake-gv-camera",
]


def find_binary() -> str:
    for name in _BINARY_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    raise SystemExit(
        "arv-fake-gv-camera not found on PATH.\n"
        "It ships with the Aravis tools package (e.g. `apt install aravis-tools`)."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run N Aravis fake GigE cameras")
    parser.add_argument("count", type=int, nargs="?", default=20,
                        help="How many fake cameras to start (default 20)")
    parser.add_argument("--interface", default="127.0.0.1",
                        help="Interface address to bind (default 127.0.0.1)")
    args = parser.parse_args()

    binary = find_binary()
    procs: List[subprocess.Popen] = []

    print(f"Starting {args.count} fake camera(s) using {binary} …")
    for i in range(args.count):
        serial = f"FAKE{i:03d}"
        cmd = [binary, "-s", serial, "-i", args.interface]
        try:
            procs.append(subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        except Exception as exc:
            print(f"  failed to start camera {serial}: {exc}", file=sys.stderr)
    print(f"{len(procs)} camera(s) running. Press Ctrl-C to stop.")

    def _shutdown(_sig, _frm):
        print("\nStopping fake cameras …")
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        time.sleep(1)
        if all(p.poll() is not None for p in procs):
            print("All fake cameras exited.")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())

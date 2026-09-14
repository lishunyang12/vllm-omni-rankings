#!/usr/bin/env python3
"""Build the revised full-VAE report from checked-in evidence, without GPUs.

The original 71-page generator is retained in archive-20260907/.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "update-20260914" / "generate_revision.py"),
        run_name="__main__",
    )

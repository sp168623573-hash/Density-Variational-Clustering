"""Locate existing baseline inputs and executables for the source package."""

import os
from pathlib import Path


LOCAL = Path(__file__).resolve().parent
PAPER = Path(os.environ.get("PAPER_EXPERIMENT_ROOT", r"G:\Experiments\paper\software paper"))
ARCHIVE = PAPER / "base line"
RECIPES = ARCHIVE / "revision_20261102"
DATA = LOCAL / "data" if (LOCAL / "data/DATA_MANIFEST.json").is_file() else ARCHIVE / "data"


def native_tool(name):
    local = LOCAL / "bin" / name
    return local if local.is_file() else ARCHIVE / "bin" / name

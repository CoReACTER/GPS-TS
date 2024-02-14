import os
from glob import glob
from pathlib import Path


def process_bh9(
    xyz_dir: str | Path
):
    if isinstance(xyz_dir, str):
        xyz_dir = Path(xyz_dir)

    ts_files = glob("*TS.xyz", root_dir=xyz_dir)

    
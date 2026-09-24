from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "pandas",
    "PIL",
    "skimage",
    "matplotlib",
    "medpy",
]


def check_package(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ModuleNotFoundError:
        print(f"Missing dependency: {name}")
        return False


def main() -> int:
    print("Checking project environment...\n")

    missing = []
    for package in REQUIRED_PACKAGES:
        if not check_package(package):
            missing.append(package)

    data_dir = PROJECT_ROOT / "Data"
    if data_dir.exists():
        print(f"Dataset folder found: {data_dir}")
    else:
        print(f"Dataset folder not found: {data_dir} (expected for training/inference)")

    print("\nSummary:")
    if missing:
        print(f"{len(missing)} required package(s) missing.")
        print("Install them with: python -m pip install -r requirements.txt")
        return 1

    print("Environment looks ready for the ACDC segmentation project.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

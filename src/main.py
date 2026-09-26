"""
MASTER PIPELINE
Runs the handwritten process in sequence.
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC = BASE_DIR / "src"


def run_step(filename):
    print(f"\n{'=' * 60}")
    print(f"Running: {filename}")
    print(f"{'=' * 60}")

    script = SRC / filename

    if not script.exists():
        print(f"ERROR: File not found: {script}")
        return False

    try:
        subprocess.run(
            [sys.executable, str(script)],
            check=True
        )
        print(f"SUCCESS: {filename}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"FAILED: {filename}")
        print(f"Exit code: {e.returncode}")
        return False


def main():

    steps = [
        "01_load_data.py",
        "02_preprocess.py",
        "03_eda.py",
        "04_feature_engineering.py",
        "05_model_building.py",
        "06_model_comparison.py",
        "07_cross_validation_tuning.py",
        "08_deployment.py",
        "09_final_model.py",
        "10_prediction.py",
        "11_inference.py",
        "12_export_model.py",
        "13_test_evaluation.py",
    ]

    for filename in steps:
        success = run_step(filename)

        if not success:
            print(f"\nPipeline stopped at: {filename}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("FULL HEALTHCARE CAPSTONE PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
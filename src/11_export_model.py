"""
12 - TRAINED MODEL EXPORT / PREPARATION
Exports a deployment-ready copy of the final trained model.
"""

import shutil
from pathlib import Path
from config import MODELS

def main():
    source = MODELS / "final_model.joblib"
    export_dir = MODELS / "deployment"
    export_dir.mkdir(exist_ok=True)

    target = export_dir / "healthcare_provider_utilization_model.joblib"

    if not source.exists():
        raise FileNotFoundError("Run 08_final_model.py before exporting.")

    shutil.copy2(source, target)

    print("TRAINED MODEL EXPORT COMPLETE")
    print(f"Deployment artifact: {target}")

if __name__ == "__main__":
    main()

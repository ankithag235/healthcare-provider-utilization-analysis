"""
08 - FINAL MODEL
Selects the best tuned model and saves the final model artifact.
"""

from pathlib import Path
import shutil
from config import MODELS

def main():
    source = MODELS / "best_model_cv_tuned.joblib"
    target = MODELS / "final_model.joblib"

    if not source.exists():
        raise FileNotFoundError("Run 07_cross_validation_tuning.py first.")

    shutil.copy2(source, target)

    print("FINAL MODEL READY")
    print(f"Saved: {target}")

if __name__ == "__main__":
    main()

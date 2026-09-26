"""
11 - INFERENCE
Loads the exported model and demonstrates inference on a supplied CSV.

Usage:
python src/11_inference.py path/to/new_data.csv
"""

import sys
import joblib
import pandas as pd
from config import MODELS

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/11_inference.py path/to/new_data.csv")
        return

    input_path = sys.argv[1]
    model = joblib.load(MODELS / "final_model.joblib")
    new_data = pd.read_csv(input_path)

    predictions = model.predict(new_data)
    output = new_data.copy()
    output["predicted_utilization"] = predictions

    print("INFERENCE COMPLETE")
    print(output[["predicted_utilization"]].head(20).to_string(index=False))

    output.to_csv("inference_predictions.csv", index=False)

if __name__ == "__main__":
    main()

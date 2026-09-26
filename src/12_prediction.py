"""
09 - PREDICTION
Runs predictions using the final trained model.
"""

import joblib
import pandas as pd
from config import PROCESSED_DATA, MODELS, TABLES

def main():
    model_path = MODELS / "final_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Run 08_final_model.py first.")

    X_test = pd.read_csv(PROCESSED_DATA / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DATA / "y_test.csv").iloc[:, 0].astype(str)

    model = joblib.load(model_path)
    predictions = model.predict(X_test)

    result = X_test.copy()
    result["actual_utilization"] = y_test.values
    result["predicted_utilization"] = predictions

    result.to_csv(TABLES / "final_predictions.csv", index=False)

    print("PREDICTION COMPLETE")
    print(result[["actual_utilization", "predicted_utilization"]].head(20).to_string(index=False))

if __name__ == "__main__":
    main()

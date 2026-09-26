"""
06 - COMPARISON OF MODELS
Compares baseline models on the same held-out test split.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from config import PROCESSED_DATA, TABLES

def main():
    X_test = pd.read_csv(PROCESSED_DATA / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DATA / "y_test.csv").iloc[:, 0].astype(str)

    model_files = {
        "Logistic Regression": Path("models/logistic_regression_baseline.joblib"),
        "Decision Tree": Path("models/decision_tree_baseline.joblib"),
        "Random Forest": Path("models/random_forest_baseline.joblib"),
    }

    rows = []
    for name, path in model_files.items():
        model = joblib.load(path)
        pred = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="weighted", zero_division=0
        )
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
        })

    results = pd.DataFrame(rows).sort_values("f1_weighted", ascending=False)
    results.to_csv(TABLES / "model_comparison.csv", index=False)

    print("MODEL COMPARISON")
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()

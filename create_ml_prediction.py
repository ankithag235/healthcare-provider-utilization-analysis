import pandas as pd
import joblib
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent

X_test_path = PROJECT_DIR / "data" / "processed" / "X_test.csv"
y_test_path = PROJECT_DIR / "data" / "processed" / "y_test.csv"
model_path = PROJECT_DIR / "models" / "final_model.joblib"

output_path = PROJECT_DIR / "data" / "processed" / "ML_Prediction_Table.xlsx"

# Load test data
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)

# Load trained model
model = joblib.load(model_path)

# Generate predictions
y_pred = model.predict(X_test)

# Create prediction table
prediction_table = X_test.copy()

# Add actual and predicted values
prediction_table["Actual_Utilization"] = y_test.iloc[:, 0].values
prediction_table["Predicted_Utilization"] = y_pred

# Check whether prediction is correct
prediction_table["Prediction_Correct"] = (
    prediction_table["Actual_Utilization"]
    == prediction_table["Predicted_Utilization"]
).map({
    True: "Yes",
    False: "No"
})

# Save Excel
prediction_table.to_excel(
    output_path,
    index=False
)

print("\nML Prediction Table created successfully!")
print(f"Saved to: {output_path}")

print("\nFirst 5 predictions:")
print(prediction_table[
    [
        "Actual_Utilization",
        "Predicted_Utilization",
        "Prediction_Correct"
    ]
].head())
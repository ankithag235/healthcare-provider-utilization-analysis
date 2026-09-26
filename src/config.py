from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "healthcare_providers_capstone1.csv"
PROCESSED_DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
MODELS = ROOT / "models"

for path in [PROCESSED_DATA, REPORTS, FIGURES, TABLES, MODELS]:
    path.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TARGET = "utilization_category"

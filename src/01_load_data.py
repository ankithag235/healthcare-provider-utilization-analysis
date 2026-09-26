"""
01 - DATA / SELECT / RAW DATA
Loads the actual CSV and creates a raw-data snapshot.
"""

from pathlib import Path
import pandas as pd
from config import RAW_DATA, PROCESSED_DATA
from utils import clean_columns

def main():
    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA}\n"
            "Place the actual CSV in data/raw/."
        )

    df = pd.read_csv(RAW_DATA, low_memory=False)
    df = clean_columns(df)

    df.to_csv(PROCESSED_DATA / "raw_data_snapshot.csv", index=False)

    print("RAW DATA LOADED")
    print("=" * 60)
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")
    print("\nColumns:")
    for c in df.columns:
        print(f" - {c}")

if __name__ == "__main__":
    main()

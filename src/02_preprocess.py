"""
02 - PREPROCESSING
Cleans columns, duplicates, numeric fields and basic missing values.
"""

import pandas as pd
import numpy as np
from config import RAW_DATA, PROCESSED_DATA
from utils import clean_columns, find_col, numericize

NUMERIC_ALIASES = [
    ["number_of_services", "services"],
    ["number_of_medicare_beneficiaries", "medicare_beneficiaries"],
    ["number_of_distinct_medicare_beneficiary_per_day_services"],
    ["average_medicare_allowed_amount"],
    ["average_submitted_charge_amount"],
    ["average_medicare_payment_amount"],
    ["average_medicare_standardized_amount"],
]

def main():
    df = pd.read_csv(RAW_DATA, low_memory=False)
    df = clean_columns(df)

    before = len(df)
    df = df.drop_duplicates().copy()

    numeric_cols = []
    for aliases in NUMERIC_ALIASES:
        col = find_col(df, aliases)
        if col:
            numeric_cols.append(col)

    df = numericize(df, numeric_cols)

    # Keep missing categorical values explicit rather than silently deleting records.
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    # Median imputation for numeric fields.
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df.to_csv(PROCESSED_DATA / "cleaned_data.csv", index=False)

    print("PREPROCESSING COMPLETE")
    print(f"Rows before duplicate removal : {before:,}")
    print(f"Rows after duplicate removal  : {len(df):,}")
    print(f"Numeric fields processed       : {len(numeric_cols)}")

if __name__ == "__main__":
    main()

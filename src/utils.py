import re
import numpy as np
import pandas as pd

def clean_column_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

def clean_columns(df):
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]
    return df

def find_col(df, candidates):
    normalized = {clean_column_name(c): c for c in df.columns}
    for candidate in candidates:
        c = clean_column_name(candidate)
        if c in normalized:
            return normalized[c]
    for candidate in candidates:
        c = clean_column_name(candidate)
        for normalized_name, original in normalized.items():
            if c in normalized_name or normalized_name in c:
                return original
    return None

def numericize(df, columns):
    df = df.copy()
    for col in columns:
        if col and col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip(),
                errors="coerce"
            )
    return df

def safe_divide(a, b):
    return a / b.replace(0, np.nan)

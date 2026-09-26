"""
04 - FEATURE ENGINEERING
Creates meaningful variables and the Low/Medium/High utilization target.
"""

import pandas as pd
import numpy as np
from config import PROCESSED_DATA
from utils import find_col, safe_divide

def main():
    input_file = PROCESSED_DATA / "cleaned_data.csv"
    output_file = PROCESSED_DATA / "features.csv"

    df = pd.read_csv(input_file)

    services = find_col(df, ["number_of_services", "services"])
    beneficiaries = find_col(df, ["number_of_medicare_beneficiaries", "medicare_beneficiaries"])
    allowed = find_col(df, ["average_medicare_allowed_amount"])
    submitted = find_col(df, ["average_submitted_charge_amount"])
    payment = find_col(df, ["average_medicare_payment_amount"])

    if not services or not beneficiaries:
        raise ValueError("Services and Medicare beneficiaries are required for feature engineering.")

    df["service_per_beneficiary"] = safe_divide(df[services], df[beneficiaries])
    df["utilization_rate"] = df["service_per_beneficiary"]

    if payment:
        df["payment_per_service"] = safe_divide(df[payment], df[services])

    if allowed and payment:
        df["payment_gap"] = df[allowed] - df[payment]

    if submitted and payment:
        df["charge_gap"] = df[submitted] - df[payment]
        df["charge_to_payment_ratio"] = safe_divide(df[submitted], df[payment])

    # Default target methodology: tertiles of service volume.
    q1, q2 = df[services].quantile([1/3, 2/3])
    df["utilization_category"] = pd.cut(
        df[services],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    df.to_csv(output_file, index=False)

    print("FEATURE ENGINEERING COMPLETE")
    print(f"Low threshold    : {q1}")
    print(f"High threshold   : {q2}")
    print("Target: utilization_category = Low / Medium / High")

if __name__ == "__main__":
    main()

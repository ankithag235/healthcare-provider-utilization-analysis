"""
03 - EDA / INFERENCE
Creates descriptive statistics, correlation data and business-analysis tables.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import PROCESSED_DATA, TABLES, FIGURES
from utils import find_col

def main():
    df = pd.read_csv(PROCESSED_DATA / "cleaned_data.csv")

    # Descriptive statistics
    df.describe(include="all").transpose().to_csv(TABLES / "descriptive_statistics.csv")

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        numeric.corr(numeric_only=True).to_csv(TABLES / "correlation_matrix.csv")

        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric.corr(numeric_only=True), cmap="Blues", center=0)
        plt.title("Healthcare Numeric Feature Correlation")
        plt.tight_layout()
        plt.savefig(FIGURES / "correlation_heatmap.png", dpi=180)
        plt.close()

    provider_type = find_col(df, ["provider_type"])
    services = find_col(df, ["number_of_services", "services"])
    state = find_col(df, ["state_code", "provider_state"])
    hcpcs = find_col(df, ["hcpcs_code"])

    if provider_type and services:
        table = (df.groupby(provider_type)[services]
                 .sum()
                 .sort_values(ascending=False)
                 .head(20))
        table.to_csv(TABLES / "provider_type_service_volume.csv")

        plt.figure(figsize=(11, 6))
        table.sort_values().plot(kind="barh")
        plt.title("Top Provider Types by Service Volume")
        plt.xlabel("Total Services")
        plt.tight_layout()
        plt.savefig(FIGURES / "provider_type_service_volume.png", dpi=180)
        plt.close()

    if state and services:
        table = (df.groupby(state)[services]
                 .sum()
                 .sort_values(ascending=False)
                 .head(20))
        table.to_csv(TABLES / "state_service_volume.csv")

    if hcpcs and services:
        table = (df.groupby(hcpcs)[services]
                 .sum()
                 .sort_values(ascending=False)
                 .head(20))
        table.to_csv(TABLES / "top_hcpcs_service_volume.csv")

    print("EDA COMPLETE")
    print("Review reports/tables and reports/figures for actual findings.")

if __name__ == "__main__":
    main()

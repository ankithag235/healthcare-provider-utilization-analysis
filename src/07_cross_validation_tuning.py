
"""
07 - CROSS VALIDATION + FINE-TUNING
Tunes Random Forest with cross-validation.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import PROCESSED_DATA, MODELS, RANDOM_STATE, CV_FOLDS
from utils import find_col

def main():
    df = pd.read_csv(PROCESSED_DATA / "features.csv").dropna(subset=["utilization_category"])
    target = "utilization_category"

    candidate = [
        find_col(df, ["provider_type"]),
        find_col(df, ["gender"]),
        find_col(df, ["entity_type"]),
        find_col(df, ["state_code", "provider_state"]),
        find_col(df, ["place_of_service"]),
        find_col(df, ["number_of_services", "services"]),
        find_col(df, ["number_of_medicare_beneficiaries", "medicare_beneficiaries"]),
        "service_per_beneficiary",
        "payment_per_service",
        find_col(df, ["average_medicare_allowed_amount"]),
        find_col(df, ["average_medicare_payment_amount"]),
        "charge_to_payment_ratio",
    ]
    features = [c for c in candidate if c and c in df.columns]

    X = df[features]
    y = df[target].astype(str)

    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    pipeline = Pipeline([
        ("preprocessor", pre),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    ])

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    }

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=CV_FOLDS,
        n_jobs=-1,
        verbose=3
    )
    search.fit(X, y)

    joblib.dump(search.best_estimator_, MODELS / "best_model_cv_tuned.joblib")

    pd.DataFrame(search.cv_results_).sort_values(
        "rank_test_score"
    ).head(20).to_csv(
        Path("reports/tables/cross_validation_results.csv"), index=False
    )

    print("CROSS VALIDATION + FINE-TUNING COMPLETE")
    print("Best parameters:", search.best_params_)
    print("Best CV F1:", search.best_score_)

if __name__ == "__main__":
    main()

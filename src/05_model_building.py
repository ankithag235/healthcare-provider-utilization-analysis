"""
05 - MODEL BUILDING
Builds baseline Logistic Regression, Decision Tree and Random Forest models.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from config import PROCESSED_DATA, RANDOM_STATE, TEST_SIZE
from utils import find_col

def get_data():
    df = pd.read_csv(PROCESSED_DATA / "features.csv")
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
    df = df.dropna(subset=[target]).copy()

    return df, features, target

def build_preprocessor(X):
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])
    return preprocessor

def main():
    df, features, target = get_data()
    X = df[features]
    y = df[target].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    trained = {}

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", estimator)
        ])
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline

    # Save split data for later consistent evaluation.
    X_train.to_csv(PROCESSED_DATA / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA / "y_test.csv", index=False)

    import joblib
    for name, model in trained.items():
        filename = name.lower().replace(" ", "_") + "_baseline.joblib"
        joblib.dump(model, Path("models") / filename)

    print("MODEL BUILDING COMPLETE")
    print("Models:", ", ".join(models.keys()))

if __name__ == "__main__":
    from pathlib import Path
    main()

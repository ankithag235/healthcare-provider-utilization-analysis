import os
import joblib
import pandas as pd


# ============================================================
# 1. MODEL PATH
# ============================================================

MODEL_PATH = "models/best_model_cv_tuned.joblib"

# If you want to use your tuned model instead, use:
# MODEL_PATH = "models/best_model_cv_tuned.joblib"


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("Loading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")
print("Model type:", type(model))


# ============================================================
# 3. CHECK IF MODEL IS A PIPELINE
# ============================================================

estimator = model

if hasattr(model, "named_steps"):

    print("\nPipeline detected.")

    print("\nPipeline steps:")

    for name, step in model.named_steps.items():
        print(f"{name}: {type(step)}")

    # Get final model
    estimator = model.steps[-1][1]

    print("\nFinal estimator:")
    print(type(estimator))


# ============================================================
# 4. CHECK FEATURE IMPORTANCE
# ============================================================

if not hasattr(estimator, "feature_importances_"):

    print("\nERROR:")
    print("The model does not contain feature_importances_.")

    print("\nThis model may not be a Random Forest or another")
    print("tree-based model.")

    raise AttributeError(
        "feature_importances_ is not available in this model."
    )


# Get importance values

importance_values = estimator.feature_importances_

print("\nNumber of importance values:")
print(len(importance_values))


# ============================================================
# 5. GET FEATURE NAMES
# ============================================================

feature_names = None


# ------------------------------------------------------------
# OPTION A: Pipeline preprocessing
# ------------------------------------------------------------

if hasattr(model, "get_feature_names_out"):

    try:

        feature_names = model[:-1].get_feature_names_out()

        print("\nFeature names obtained from preprocessing pipeline.")

    except Exception as e:

        print("\nCould not get pipeline feature names:")
        print(e)


# ------------------------------------------------------------
# OPTION B: Model feature names
# ------------------------------------------------------------

if feature_names is None:

    if hasattr(estimator, "feature_names_in_"):

        feature_names = estimator.feature_names_in_

        print("\nFeature names obtained from model.")


# ============================================================
# 6. IF FEATURE NAMES ARE NOT FOUND
# ============================================================

if feature_names is None:

    print("\nERROR:")
    print("Feature names could not be detected automatically.")

    print("\nModel attributes:")

    print(dir(estimator))

    raise ValueError(
        "Feature names are not available automatically."
    )


# ============================================================
# 7. CHECK FEATURE COUNT
# ============================================================

print("\nFeature count:")
print("Feature names:", len(feature_names))
print("Importance values:", len(importance_values))


if len(feature_names) != len(importance_values):

    raise ValueError(
        f"""
Feature count mismatch!

Feature names = {len(feature_names)}
Importance values = {len(importance_values)}

Your model probably contains preprocessing such as
OneHotEncoder or ColumnTransformer.
"""
    )


# ============================================================
# 8. CREATE FEATURE IMPORTANCE DATAFRAME
# ============================================================

feature_importance = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance_values

})


# ============================================================
# 9. SORT IMPORTANCE
# ============================================================

feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

).reset_index(drop=True)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(feature_importance.to_string(index=False))


# ============================================================
# 11. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_FOLDER = "data/processed"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# 12. SAVE CSV
# ============================================================

OUTPUT_FILE = os.path.join(

    OUTPUT_FOLDER,

    "feature_importance.csv"

)


feature_importance.to_csv(

    OUTPUT_FILE,

    index=False

)


# ============================================================
# 13. SUCCESS MESSAGE
# ============================================================

print("\n")
print("=" * 60)
print("SUCCESS!")
print("=" * 60)

print("\nFeature importance file created:")

print(OUTPUT_FILE)

print("\nColumns created:")

print("Feature")
print("Importance")

print("\nYou can now import this CSV into Power BI.")
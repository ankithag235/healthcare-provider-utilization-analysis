# Healthcare Provider Performance, Medicare Cost & Utilization Analytics with Machine Learning

## VS Code project flow — matching the handwritten format

`# 🔗 Data Preparation

The healthcare dataset is loaded and prepared before exploratory analysis and machine learning.

The data preparation workflow is:

```text
Raw Healthcare Data
        ↓
Load Dataset
        ↓
Inspect Structure
        ↓
Validate Data Types
        ↓
Handle Missing Values
        ↓
Remove / Check Duplicates
        ↓
Validate Numeric Variables
        ↓
Check Outliers
        ↓
Prepare Analytical Dataset
```

## Data Loading

The raw healthcare provider dataset is loaded using Pandas.

```python
import pandas as pd

df_raw = pd.read_csv(
    "data/raw/healthcare_providers_capstone1.csv.csv"
)
```

The initial dataset is inspected using:

```python
df_raw.shape
df_raw.head()
df_raw.info()
df_raw.describe()
```

This provides an understanding of the number of observations, columns, data types, and numerical distributions.

---

## Data Structure Validation

The following checks are performed:

```python
print("Rows:", df_raw.shape[0])
print("Columns:", df_raw.shape[1])

print(df_raw.dtypes)

print(df_raw.isnull().sum())
```

These checks help identify:

* Dataset dimensions.
* Numerical and categorical columns.
* Missing values.
* Incorrect or unexpected data types.
* Variables requiring preprocessing.

---

## Missing Value Analysis

Missing values are analyzed before model development.

Example:

```python
missing_values = (
    df_raw.isnull()
    .sum()
    .sort_values(ascending=False)
)

missing_percentage = (
    df_raw.isnull()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)
```

The analysis identifies columns with substantial missingness.

Examples of variables that may contain missing values include:

```text
Street Address 2
Middle Initial
Credentials
First Name
Gender
```

Missing-value treatment is performed according to the nature of each variable.

For categorical variables, appropriate categorical handling or an `"Unknown"` category can be used where applicable.

For numerical variables, imputation or other appropriate treatment should be applied based on the modeling pipeline.

---

## Duplicate Analysis

Duplicate records are checked before modeling.

```python
duplicate_count = df_raw.duplicated().sum()

print("Duplicate rows:", duplicate_count)
```

Duplicates are reviewed before removal because healthcare claims/service datasets may legitimately contain repeated provider or service combinations.

Therefore, duplicate handling should be based on the business meaning of the record rather than automatically deleting every repeated row.

---

## Data Type Conversion

Numerical healthcare variables are converted to appropriate numeric data types where required.

Examples include:

```text
Number of Services
Medicare Beneficiaries
Distinct Beneficiary/Day Services
Average Allowed Amount
Average Submitted Charge
Average Medicare Payment
Average Standardized Payment
```

Example:

```python
numeric_columns = [
    "Number of Services",
    "Medicare Beneficiaries",
    "Distinct Beneficiary/Day Services",
    "Average Allowed Amount",
    "Average Submitted Charge",
    "Average Medicare Payment",
    "Average Standardized Payment"
]

for col in numeric_columns:
    df_raw[col] = pd.to_numeric(
        df_raw[col],
        errors="coerce"
    )
```

---

# 🔎 Exploratory Data Analysis

The EDA stage examines provider utilization, service activity, beneficiaries, Medicare payments, geographic patterns, and relationships between numerical variables.

The main areas of analysis are:

```text
1. Provider Analysis
2. Service Analysis
3. Beneficiary Analysis
4. Medicare Payment Analysis
5. Geographic Analysis
6. Correlation Analysis
7. Utilization Analysis
```

---

## Provider Analysis

Provider activity is analyzed using:

```text
Provider Type
State
Place of Service
Entity Type
```

Key business questions include:

* Which provider types have the highest service volume?
* Which states have the highest provider activity?
* Which provider categories serve the largest number of beneficiaries?
* Which provider types have higher Medicare payments?

---

## Service Analysis

Healthcare service utilization is analyzed using:

```text
HCPCS Code
HCPCS Description
Number of Services
Medicare Beneficiaries
```

The analysis identifies:

* High-volume healthcare services.
* Services associated with large beneficiary populations.
* Frequently performed procedures.
* Service-level utilization patterns.

---

## Medicare Payment Analysis

Medicare financial measures are analyzed using:

```text
Average Allowed Amount
Average Submitted Charge
Average Medicare Payment
Average Standardized Payment
```

The analysis investigates:

```text
Submitted Charge
        ↓
Allowed Amount
        ↓
Medicare Payment
        ↓
Standardized Payment
```

This helps understand differences between submitted charges and Medicare payments.

---

## Beneficiary Analysis

Beneficiary-related variables are analyzed to understand provider utilization.

Important variables include:

```text
Medicare Beneficiaries
Distinct Beneficiary/Day Services
Number of Services
```

Relationships such as:

```text
Services vs Beneficiaries
```

are examined to identify providers with relatively high service activity.

---

## Geographic Analysis

Provider utilization is analyzed across states.

```text
State
  ↓
Provider Count
  ↓
Service Volume
  ↓
Beneficiaries
  ↓
Medicare Payment
```

This allows geographic differences in healthcare utilization and Medicare spending to be explored.

---

# 🛠️ Feature Engineering

Feature engineering transforms the raw healthcare variables into useful analytical and machine-learning features.

The objective is to create variables that describe:

```text
Provider Activity
Service Utilization
Beneficiary Utilization
Payment Intensity
```

---

## Payment Per Service

A derived metric can be calculated as:

```text
Payment Per Service
=
Medicare Payment / Number of Services
```

Example:

```python
df["Payment_Per_Service"] = (
    df["Average Medicare Payment"] /
    df["Number of Services"].replace(0, pd.NA)
)
```

---

## Services Per Beneficiary

A service utilization ratio can be calculated as:

```text
Service Per Beneficiary
=
Number of Services / Medicare Beneficiaries
```

Example:

```python
df["Service_Per_Beneficiary"] = (
    df["Number of Services"] /
    df["Medicare Beneficiaries"].replace(0, pd.NA)
)
```

This provides an additional measure of provider service intensity.

---

## Charge-to-Payment Relationship

Where appropriate, a charge/payment ratio can be derived to examine the relationship between submitted charges and Medicare payments.

```text
Charge-to-Payment Ratio
=
Submitted Charge / Medicare Payment
```

This variable should be interpreted carefully because the underlying amounts may represent averages rather than total financial values.

---

# 🎯 Utilization Category

The machine-learning problem is formulated as a classification task.

The target variable is:

```text
Utilization_Category
```

Providers are categorized into:

```text
Low
Medium
High
```

Conceptually:

```text
Provider Utilization
        ↓
Utilization Measure
        ↓
Classification Threshold
        ↓
┌────────┬─────────┬────────┐
│  Low   │ Medium  │  High  │
└────────┴─────────┴────────┘
```

The exact threshold methodology should remain consistent with the methodology implemented in the project notebook.

---

# 🧩 Feature Selection

The machine-learning dataset uses provider, geographic, service, beneficiary, and payment characteristics.

The selected features should be documented according to the final model implementation.

A representative feature structure is:

```python
features = [
    "Provider Type",
    "State",
    "Place of Service",
    "HCPCS Code",
    "Drug Indicator",
    "Medicare Beneficiaries",
    "Average Allowed Amount",
    "Average Submitted Charge",
    "Average Medicare Payment"
]
```

Target:

```python
target = "Utilization_Category"
```

The input matrix is:

```python
X = df[features]
```

and the target vector is:

```python
y = df[target]
```

---

# ✂️ Train / Test Split

The dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

Example:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The test dataset remains unseen during model training and hyperparameter selection.

`stratify=y` helps maintain the distribution of utilization categories between the training and testing datasets.

---

# 📏 Feature Preprocessing

The healthcare dataset contains both numerical and categorical variables.

Therefore, separate preprocessing strategies are applied.

## Numerical Features

Numerical variables can be standardized using:

```text
StandardScaler
```

Examples:

```text
Medicare Beneficiaries
Average Allowed Amount
Average Submitted Charge
Average Medicare Payment
```

Standardization follows:

```text
z = (x - μ) / σ
```

where:

* `x` = original value
* `μ` = training-data mean
* `σ` = training-data standard deviation
* `z` = standardized value

The scaler is fitted only on training data.

```python
X_train_scaled = scaler.fit_transform(X_train_numeric)
```

The same transformation is applied to the test data:

```python
X_test_scaled = scaler.transform(X_test_numeric)
```

---

## Categorical Features

Categorical variables such as:

```text
Provider Type
State
Place of Service
HCPCS Code
Drug Indicator
```

can be transformed using:

```text
OneHotEncoder
```

This converts categorical values into machine-readable numerical representations.

---

# 🤖 Model Building and Comparison

Because the target is:

```text
Low / Medium / High
```

the project uses classification algorithms.

The models evaluated can include:

```text
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
```

The models are trained using the processed training dataset.

The purpose of model comparison is to evaluate different approaches using the same training/testing methodology.

---

# 📊 Classification Evaluation Metrics

The models are evaluated using multiple classification metrics.

## Accuracy

Accuracy measures the proportion of correctly classified observations.

```text
Accuracy
=
Correct Predictions
-------------------
Total Predictions
```

---

## Precision

Precision measures how many observations predicted as a particular class were actually members of that class.

```text
Precision
=
True Positives
---------------------------
True Positives + False Positives
```

---

## Recall

Recall measures how many actual observations belonging to a class were correctly identified.

```text
Recall
=
True Positives
-------------------------
True Positives + False Negatives
```

---

## F1-Score

F1-score combines precision and recall.

```text
F1 =
2 × Precision × Recall
----------------------
Precision + Recall
```

For a multi-class utilization problem, macro and/or weighted averages can be reported depending on the evaluation methodology.

---

# 📋 Confusion Matrix

A confusion matrix is used to examine classification errors.

Conceptually:

```text
                 Predicted
              Low Medium High

Actual Low      ✓    •     •

Actual Medium  •    ✓     •

Actual High    •    •     ✓
```

The confusion matrix helps identify which utilization categories are being confused by the model.

---

# 🎛️ Cross Validation and Hyperparameter Tuning

After baseline model comparison, the selected classification model can be further optimized using cross-validation and hyperparameter tuning.

The process is:

```text
Training Data
      ↓
Cross Validation
      ↓
Parameter Grid
      ↓
GridSearchCV
      ↓
Candidate Models
      ↓
Best Parameters
      ↓
Final Tuned Model
```

For example, a Random Forest classifier can be tuned using parameters such as:

```python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
```

The exact parameter grid should match the implementation used in the project.

---

# 🧪 Final Test Evaluation

After hyperparameter tuning, the final model is evaluated against the previously unseen test dataset.

The evaluation includes:

```text
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
```

The final results should be reported using the actual values produced by the project notebook.

```text
Final Model:
[Actual Model]

Accuracy:
[Actual Result]

Precision:
[Actual Result]

Recall:
[Actual Result]

F1-Score:
[Actual Result]
```

---

# 📈 Feature Importance

Feature importance is used to understand which variables contribute to utilization classification.

For tree-based models, feature importance can be extracted from the trained model.

Conceptually:

```text
Feature Importance
        ↓
Number of Services
        ↓
Medicare Beneficiaries
        ↓
Payment Per Service
        ↓
Service Per Beneficiary
        ↓
Other Provider / Service Features
```

The feature-importance analysis helps connect the machine-learning results with healthcare business questions.

---

# 💾 Model Artifact Export

The trained preprocessing pipeline and final model can be serialized for reuse.

Example artifacts:

```text
data_preprocessor.pkl
best_model_cv_tuned.joblib
healthcare_provider_utilization_model.joblib
```

These artifacts allow the trained machine-learning workflow to be reused without retraining the model from scratch.

---

# 📤 Prediction Export

Predictions from the final model are exported for downstream analysis.

Example:

```text
healthcare_provider_predictions.csv
```

The prediction dataset can contain:

```text
Provider Information
Service Information
Beneficiary Information
Payment Information
Actual Utilization Category
Predicted Utilization Category
```

Conceptually:

```text
Test Data
    ↓
Preprocessing
    ↓
Final Model
    ↓
Predicted Utilization
    ↓
CSV Export
    ↓
Power BI
```

---

# 📊 Power BI Dashboard

The prediction and analytical data are loaded into Power BI to create an interactive healthcare analytics dashboard.

The dashboard focuses on:

```text
Provider Performance
Service Utilization
Medicare Cost Analysis
Geographic Analysis
Machine Learning Insights
```

---

# 📌 Power BI Page 1 — Executive Overview

The Executive Overview provides high-level KPIs.

Recommended KPIs include:

```text
Total Providers
Total Services
Total Beneficiaries
Total Medicare Payment
Average Payment
High-Utilization Providers
```

The page provides a high-level summary of provider utilization and Medicare payment activity.

---

# 📌 Power BI Page 2 — Provider Performance

This page analyzes provider-level and provider-type performance.

Key dimensions include:

```text
Provider
Provider Type
State
Number of Services
Beneficiaries
Medicare Payment
```

Recommended visuals:

```text
Provider Type → Service Volume

State → Provider Activity

Provider → Medicare Payment

Provider → Beneficiaries
```

Slicers can be provided for:

```text
State
Provider Type
Place of Service
```

---

# 📌 Power BI Page 3 — Medicare Cost & Service Analysis

This page focuses on the relationship between healthcare services, beneficiaries, and Medicare payments.

Key measures include:

```text
Number of Services
Medicare Beneficiaries
Average Allowed Amount
Average Submitted Charge
Average Medicare Payment
Average Standardized Payment
```

Recommended visuals include:

```text
Service Volume vs Medicare Payment

Beneficiaries vs Services

Submitted Charge vs Medicare Payment

Top HCPCS Services

Medicare Payment by Provider Type

Medicare Payment by State
```

---

# 📌 Power BI Page 4 — ML Insights

The ML Insights page presents the output of the machine-learning model.

Recommended components include:

```text
Utilization Category Distribution
Actual vs Predicted Utilization
Confusion Matrix
Feature Importance
High-Utilization Provider Analysis
```

The utilization distribution can be displayed as:

```text
Low
Medium
High
```

The actual-versus-predicted analysis can be used to identify correctly and incorrectly classified providers.

---

# 🧱 Project Architecture

```text
                  ┌────────────────────────────┐
                  │ Healthcare Provider CSV    │
                  └──────────────┬─────────────┘
                                 ↓
                     Data Loading & Validation
                                 ↓
                          Data Cleaning
                                 ↓
                                EDA
                                 ↓
                       Feature Engineering
                                 ↓
                    Utilization Categorization
                                 ↓
                         Train / Test Split
                                 ↓
                      Feature Preprocessing
                                 ↓
              ┌──────────────────┴──────────────────┐
              ↓                                     ↓
      Logistic Regression                    Tree-Based Models
              ↓                                     ↓
              └──────────────────┬──────────────────┘
                                 ↓
                         Model Comparison
                                 ↓
                         Cross Validation
                                 ↓
                      Hyperparameter Tuning
                                 ↓
                           Final Model
                                 ↓
                      Test Set Evaluation
                                 ↓
                        Feature Importance
                                 ↓
                       Prediction Export
                                 ↓
                            Power BI
                                 ↓
                       Business Insights
```

---

# 📁 Recommended Project Structure

```text
Healthcare_Provider_Utilization/
│
├── data/
│   └── raw/
│       └── healthcare_providers_capstone1.csv.csv
│
├── notebooks/
│   └── healthcare_provider_utilization.ipynb
│
├── models/
│   ├── healthcare_provider_utilization_model.joblib
│   └── best_model_cv_tuned.joblib
│
├── outputs/
│   ├── predictions/
│   └── visualizations/
│
├── PowerBI/
│   └── healthcare_provider_utilization.pbix
│
├── requirements.txt
│
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>

cd Healthcare_Provider_Utilization
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Main Dependencies

The project uses:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Additional libraries should be included in `requirements.txt` if they are used by the final notebook.

---

# ▶️ How to Run

## Step 1 — Prepare the Dataset

Place the healthcare provider dataset in:

```text
data/raw/
```

---

## Step 2 — Open the Notebook

Open:

```text
healthcare_provider_utilization.ipynb
```

using:

```text
Jupyter Notebook
JupyterLab
VS Code
```

---

## Step 3 — Verify Dataset Path

Update the path if required:

```python
df_raw = pd.read_csv(
    "data/raw/healthcare_providers_capstone1.csv.csv"
)
```

---

## Step 4 — Run the Notebook

Run the sections in the following order:

```text
1. Problem Statement
2. Dataset & Data Dictionary
3. Data Loading
4. Data Cleaning & Validation
5. Exploratory Data Analysis
6. Feature Engineering
7. Utilization Category Creation
8. Feature Preprocessing
9. Train / Test Split
10. Model Building & Comparison
11. Cross Validation
12. Hyperparameter Tuning
13. Final Test Evaluation
14. Feature Importance
15. Prediction Export
16. Power BI Preparation
```

---

# 🔮 Using the Saved Model for New Predictions

The saved preprocessing object and trained model can be used for new provider observations.

Conceptually:

```python
import joblib

preprocessor = joblib.load(
    "data_preprocessor.pkl"
)

model = joblib.load(
    "best_model_cv_tuned.joblib"
)

new_data_processed = preprocessor.transform(
    new_data
)

predictions = model.predict(
    new_data_processed
)
```

The new data must contain the same feature columns and compatible preprocessing structure used during model training.

---

# ⚠️ Important Model Validation Note

The healthcare dataset contains variables that may be directly or indirectly related to the utilization target.

Therefore, a target-leakage audit should be performed before treating the final model performance as production-ready.

The validation process should ask:

```text
Input Feature
      ↓
Was it available before prediction?
      ↓
Was it independently measured?
      ↓
Was it used to create Utilization_Category?
      ↓
Could it directly reveal the target?
      ↓
No Target Leakage?
```

For example, if `Utilization_Category` is created directly from `Number of Services`, then using `Number of Services` as an input feature can introduce a form of target leakage or make the prediction task circular, depending on the business definition of the target.

Therefore, the target-generation methodology and feature-selection methodology should be documented together.

---

# 🔐 Reproducibility

The project uses:

```python
random_state = 42
```

where supported.

This provides reproducible train/test splits and reproducible model experiments when the same dataset, preprocessing configuration, model parameters, and software environment are used.

---

# 🧠 Key Machine Learning Concepts Demonstrated

This project demonstrates:

* Data loading and validation
* Data cleaning
* Missing-value analysis
* Duplicate analysis
* Exploratory Data Analysis
* GroupBy aggregation
* Feature engineering
* Utilization classification
* Train/test splitting
* Feature preprocessing
* Standardization
* Categorical encoding
* Classification
* Model comparison
* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Cross-validation
* Grid Search
* Hyperparameter tuning
* Feature importance
* Model serialization
* Prediction export
* Power BI visualization
* Business intelligence reporting

---

# 📌 Final Results

The final model results should be reported from the actual project execution.

```text
Final Model:
[Actual Best Model]

Best Hyperparameters:
[Actual Parameters]

Accuracy:
[Actual Result]

Precision:
[Actual Result]

Recall:
[Actual Result]

F1-Score:
[Actual Result]
```

The results should be interpreted together with the target-leakage and validation analysis.

---

# 📊 Business Value

The solution can support healthcare analytics by helping users:

* Identify high-utilization providers.
* Analyze service utilization patterns.
* Compare provider activity across states.
* Analyze Medicare payment patterns.
* Identify high-volume healthcare services.
* Examine beneficiary utilization.
* Understand service-to-payment relationships.
* Investigate utilization drivers.
* Support provider performance analysis.
* Provide interactive healthcare intelligence through Power BI.

---

# 🚀 Future Improvements

Potential improvements include:

1. Perform a formal target-leakage audit.
2. Validate utilization-category thresholds.
3. Add additional classification algorithms.
4. Perform systematic hyperparameter optimization.
5. Add class-specific performance analysis.
6. Use SHAP or permutation importance for interpretability.
7. Add model calibration where appropriate.
8. Add automated model retraining.
9. Deploy the trained model through a REST API.
10. Create an automated prediction pipeline.
11. Connect Power BI to a continuously updated prediction source.
12. Add automated alerts for high-utilization providers.
13. Incorporate additional temporal or longitudinal provider information where available.
14. Monitor model performance after deployment.

---

# 👨‍💻 Project Summary

This project demonstrates an end-to-end machine learning workflow for **Healthcare Provider Utilization & Medicare Payment Analysis**.

The system prepares healthcare provider and Medicare service data, performs exploratory analysis, engineers utilization and payment-related features, classifies providers into utilization categories, compares multiple classification algorithms, performs cross-validation and hyperparameter tuning, evaluates the final model on unseen data, extracts model insights, exports predictions, and presents the results through an interactive Power BI dashboard.

The complete pipeline is:

```text
Raw Healthcare Data
        ↓
Data Preparation
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Utilization Classification
        ↓
Machine Learning
        ↓
Model Evaluation
        ↓
Prediction
        ↓
Power BI
        ↓
Business Intelligence
```

The project combines **Python, Pandas, NumPy, Scikit-learn, Machine Learning, model evaluation, feature importance, and Power BI** to create an end-to-end healthcare analytics solution.

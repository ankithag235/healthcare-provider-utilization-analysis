
# Healthcare Provider Utilization & Medicare Payment Analysis

## 📌 Project Overview

This project analyzes healthcare provider utilization, Medicare payments, service activity, and beneficiary patterns using **Python, Machine Learning, and Power BI**.

The objective is to transform healthcare provider data into actionable analytical insights and build a machine-learning classification workflow for categorizing provider utilization into:

* Low Utilization
* Medium Utilization
* High Utilization

The project combines:

**Healthcare Data → Data Cleaning → EDA → Feature Engineering → Machine Learning → Model Evaluation → Prediction Export → Power BI Dashboard**

---

## 🎯 Business Problem

Healthcare provider datasets contain large amounts of information about:

* Provider characteristics
* Provider type
* State
* Healthcare services
* Medicare beneficiaries
* Submitted charges
* Allowed amounts
* Medicare payments
* Standardized payments

Analyzing these variables manually can make it difficult to identify utilization patterns and payment relationships.

### Business objective

Develop an analytical and machine-learning solution that can:

1. Analyze healthcare provider utilization.
2. Identify high-volume services.
3. Examine beneficiary and service relationships.
4. Analyze Medicare payment patterns.
5. Compare provider activity across states and provider types.
6. Classify providers into utilization categories.
7. Present analytical and ML insights through Power BI.

---

# 🔄 Project Workflow

```text
Business Problem
       ↓
Healthcare Provider Dataset
       ↓
Data Loading
       ↓
Data Cleaning & Validation
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Utilization Category
       ↓
Feature Preprocessing
       ↓
Train / Test Split
       ↓
Machine Learning
       ↓
Model Comparison
       ↓
Cross Validation
       ↓
Hyperparameter Tuning
       ↓
Final Model
       ↓
Feature Importance
       ↓
Prediction Export
       ↓
Power BI Dashboard
       ↓
Business Insights
```

---

# 📂 Dataset

The project uses the healthcare provider dataset:

```text
data/raw/healthcare_providers_capstone1.csv.csv
```

The dataset contains provider-level and service-level healthcare information.

### Main variables

| Category      | Variables                                 |
| ------------- | ----------------------------------------- |
| Provider      | Provider Type, Provider Name, Entity Type |
| Geography     | State, Place of Service                   |
| Service       | HCPCS Code, HCPCS Description             |
| Utilization   | Number of Services                        |
| Beneficiaries | Medicare Beneficiaries                    |
| Financial     | Average Allowed Amount                    |
| Financial     | Average Submitted Charge                  |
| Financial     | Average Medicare Payment                  |
| Financial     | Average Standardized Payment              |
| Target        | Utilization Category                      |

---

# 🧹 Data Cleaning & Preparation

The data preparation process includes:

* Dataset structure inspection
* Data type validation
* Missing-value analysis
* Duplicate analysis
* Numeric conversion
* Outlier investigation
* Preparation of the analytical dataset

### Important missing-value analysis

The project identified missing values in several provider attributes, including:

* Street Address 2
* Middle Initial
* Credentials
* First Name
* Gender

Missing-value treatment is applied according to the type and business meaning of each variable.

---

# 🔎 Exploratory Data Analysis

EDA focuses on understanding healthcare utilization and Medicare payment patterns.

### Provider Analysis

Analysis includes:

* Provider Type
* State
* Place of Service
* Entity Type

### Service Analysis

Analysis includes:

* HCPCS Code
* HCPCS Description
* Number of Services
* Medicare Beneficiaries

### Medicare Payment Analysis

Financial variables analyzed include:

* Average Allowed Amount
* Average Submitted Charge
* Average Medicare Payment
* Average Standardized Payment

### Beneficiary Analysis

Relationships between:

```text
Medicare Beneficiaries
        ↓
Number of Services
        ↓
Service Utilization
```

are analyzed to understand provider activity.

### Geographic Analysis

Provider utilization and Medicare payment patterns are examined across states.

---

# 🛠️ Feature Engineering

The project creates additional analytical features to describe provider utilization and payment intensity.

### Payment Per Service

```text
Payment Per Service =
Average Medicare Payment / Number of Services
```

### Service Per Beneficiary

```text
Service Per Beneficiary =
Number of Services / Medicare Beneficiaries
```

These derived variables help analyze service intensity and payment relationships.

---

# 🎯 Machine Learning Problem

The machine-learning task is a **multi-class classification problem**.

### Target variable

```text
Utilization_Category
```

### Classes

```text
Low
Medium
High
```

The model is designed to classify healthcare provider utilization based on provider, service, beneficiary, geographic, and payment-related characteristics.

---

# 🧩 Machine Learning Features

The project uses provider, geographic, service, beneficiary, and financial characteristics.

The documented feature set includes:

```text
Provider Type
State
Place of Service
HCPCS Code
Drug Indicator
Medicare Beneficiaries
Average Allowed Amount
Average Submitted Charge
Average Medicare Payment
```

Target:

```text
Utilization_Category
```

---

# ✂️ Train / Test Split

The machine-learning workflow uses an **80/20 train-test split** with stratification.

```text
80% → Training Data
20% → Testing Data
```

A fixed random state is used where supported:

```python
random_state = 42
```

The test data is kept separate from model training and tuning.

---

# 🤖 Machine Learning Models

The project includes baseline and tuned classification models.

### Models used

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

The trained model artifacts are stored in the project's `models` directory.

---

# 🔧 Model Tuning

The Random Forest model was further optimized using cross-validation and hyperparameter tuning.

### Cross-validation

The tuning workflow used:

```text
5-fold Cross Validation
```

with the configured parameter search producing:

```text
60 model fits
```

### Best Random Forest configuration

The best configuration identified in the project was:

| Parameter         | Value |
| ----------------- | ----: |
| n_estimators      |   100 |
| max_depth         |  None |
| min_samples_split |     2 |

The recorded best cross-validation F1 score was:

```text
Best CV F1 Score: 1.00
```

> **Important:** This is the recorded cross-validation result, not a claim about unseen real-world production performance. Because utilization is closely related to service-volume variables, the target definition and feature selection require a leakage check before interpreting this performance as production-ready.

---

# 📊 Model Evaluation

The classification workflow evaluates:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The final test-set metric values should be taken directly from the final model execution rather than estimated or manually entered.

### Current project status

| Evaluation Item       | Status                   |
| --------------------- | ------------------------ |
| Accuracy              | Evaluated in ML workflow |
| Precision             | Evaluated in ML workflow |
| Recall                | Evaluated in ML workflow |
| F1-Score              | Evaluated in ML workflow |
| Confusion Matrix      | Created for ML analysis  |
| Cross Validation      | Completed                |
| Hyperparameter Tuning | Completed                |
| Feature Importance    | Completed                |

---

# 📈 Feature Importance

The trained tree-based model provides feature-importance information.

The strongest recorded features include:

| Feature                 | Importance |
| ----------------------- | ---------: |
| Number of Services      |     0.4403 |
| Beneficiaries           |     0.2839 |
| Payment Per Service     |     0.1165 |
| Service Per Beneficiary |     0.0797 |

These features account for a large portion of the recorded model importance.

### Interpretation

The model places substantial importance on:

* Service volume
* Beneficiary volume
* Payment intensity
* Service intensity per beneficiary

These variables provide a connection between the machine-learning output and the healthcare utilization analysis.

---

# ⚠️ Target Leakage Validation

Target leakage is an important consideration in this project.

If `Utilization_Category` is created directly from `Number of Services`, then using `Number of Services` as a model input can make the prediction task circular.

For example:

```text
Number of Services
        ↓
Utilization Category
        ↓
Model also receives Number of Services
```

This can result in artificially strong model performance.

Therefore, before presenting the model as production-ready, the project should clearly document:

1. How `Utilization_Category` was created.
2. Which variables were used to create the target.
3. Which variables were provided to the model.
4. Whether those variables would be available before prediction.
5. Whether any input variable directly reveals the target.

This validation is especially important because `Number of Services` has the highest recorded feature importance.

---

# 💾 Model Artifacts

The project contains trained model artifacts, including:

```text
models/
├── best_model_cv_tuned.joblib
├── decision_tree_baseline.joblib
├── final_model.joblib
├── logistic_regression_baseline.joblib
└── random_forest_baseline.joblib
```

Processed datasets and ML outputs are maintained under:

```text
data/processed/
```

including:

```text
cleaned_data.csv
features.csv
feature_importance.csv
raw_data_snapshot.csv
X_train.csv
X_test.csv
y_train.csv
y_test.csv
```

---

# 📤 Prediction Output

The machine-learning workflow generates prediction data for downstream analysis.

The prediction workflow is:

```text
Test Data
    ↓
Preprocessing
    ↓
Trained Model
    ↓
Predicted Utilization Category
    ↓
Prediction Dataset
    ↓
Power BI
```

The prediction results are used to support the ML Insights page in Power BI.

---

# 📊 Power BI Dashboard

The Power BI dashboard converts the analytical and machine-learning results into interactive business intelligence.

The project contains a Power BI file:

```text
Healthcare Providers-capstone project .pbix
```

The dashboard is organized around four major analytical areas.

---

## 📌 Page 1 — Executive Overview

The Executive Overview provides high-level healthcare KPIs.

### Main KPIs

* Total Providers
* Total Services
* Total Beneficiaries
* Medicare Payment
* Average Payment

The page provides an overall view of provider utilization and Medicare payment activity.

---

## 📌 Page 2 — Provider Performance

This page focuses on provider-level performance.

### Analysis areas

* Provider Type
* State
* Provider Activity
* Number of Services
* Beneficiaries
* Medicare Payment

### Example analysis

```text
Provider Type → Service Volume

State → Provider Activity

Provider → Beneficiaries

Provider → Medicare Payment
```

---

## 📌 Page 3 — Medicare Cost & Service Analysis

This page examines relationships between services, beneficiaries, charges, and Medicare payments.

### Key analysis

* Service Volume
* Medicare Beneficiaries
* Average Allowed Amount
* Average Submitted Charge
* Average Medicare Payment
* Average Standardized Payment
* Service vs Payment
* Beneficiaries vs Services
* Submitted Charge vs Medicare Payment
* Payment by Provider Type
* Payment by State
* HCPCS service analysis

---

## 📌 Page 4 — ML Insights

The ML Insights page connects the machine-learning results with Power BI.

### Main visuals

* Utilization Category Distribution
* Actual vs Predicted Utilization
* Confusion Matrix
* Feature Importance
* High-Utilization Provider Analysis

### Feature Importance

The dashboard displays the important ML variables, including:

```text
Number of Services
Beneficiaries
Payment Per Service
Service Per Beneficiary
```

This allows business users to connect model results with provider utilization patterns.

---

# 🏗️ Project Architecture

```text
Healthcare Provider Dataset
          ↓
     Data Loading
          ↓
 Data Cleaning & Validation
          ↓
          EDA
          ↓
 Feature Engineering
          ↓
 Utilization Categorization
          ↓
 Feature Preprocessing
          ↓
    Train / Test Split
          ↓
   Model Development
          ↓
 Model Comparison
          ↓
 Cross Validation
          ↓
 Hyperparameter Tuning
          ↓
     Final Model
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

# 📁 Repository Structure

The current repository contains:

```text
Healthcare_Capstone_Project/
│
├── data/
├── reports/
├── sql/
├── src/
├── App/
│
├── Healthcare Providers-capstone project .pbix
├── PROBLEM_STATEMENT.md
├── PROJECT_FLOW.txt
├── create_ml_prediction.py
├── requirements.txt
├── .gitignore
└── README.md
```

The repository also contains the processed ML datasets and model-related project files under the appropriate project directories.

---

# ⚙️ Technologies Used

### Programming & Data Analysis

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Logistic Regression
* Decision Tree
* Random Forest
* Cross-validation
* Grid Search
* Hyperparameter tuning
* Classification metrics
* Feature importance

### Business Intelligence

* Power BI
* DAX
* Power Query

### Development Tools

* VS Code
* Jupyter
* Anaconda
* Git
* GitHub

---

# ▶️ How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/ankithag235/healthcare-provider-utilization-analysis.git
```

```bash
cd healthcare-provider-utilization-analysis
```

## 2. Create an Environment

Using Anaconda:

```bash
conda create -n myproject python=3.11
```

Activate:

```bash
conda activate myproject
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Open the Project

Open the project folder in VS Code.

```text
Healthcare_Capstone_Project
```

## 5. Run the Python / ML Workflow

Run the available Python scripts and notebooks in the project workflow.

The general sequence is:

```text
Data Loading
→ Cleaning
→ EDA
→ Feature Engineering
→ ML
→ Evaluation
→ Prediction Export
```

## 6. Open Power BI

Open:

```text
Healthcare Providers-capstone project .pbix
```

Then refresh the relevant data sources if required.

---

# 🔐 Reproducibility

The project uses:

```python
random_state = 42
```

where supported.

This helps reproduce the train/test split and model experiments when the same dataset, preprocessing, parameters, and environment are used.

---

# 💡 Business Insights Supported by the Project

The completed analytical workflow supports investigation of:

1. Provider utilization patterns.
2. High-volume healthcare services.
3. Beneficiary and service relationships.
4. Medicare payment patterns.
5. Differences in provider activity across states.
6. Provider-type utilization patterns.
7. Payment intensity.
8. Service intensity per beneficiary.
9. High-utilization provider groups.
10. Machine-learning drivers of utilization classification.

---

# ⚠️ Limitations

The project should be interpreted within the limitations of the available healthcare dataset.

Important considerations include:

* The dataset represents historical healthcare provider/service information.
* Utilization categories depend on the target-definition methodology.
* Some provider attributes contain missing values.
* Model performance must be interpreted in the context of the target definition.
* Target leakage must be ruled out before treating the ML model as production-ready.
* Cross-validation performance should not be treated as equivalent to real-world deployment performance.

---

# 🚀 Future Improvements

Potential next steps include:

1. Complete a formal target-leakage audit.
2. Refine utilization-category definitions if required.
3. Evaluate the final model on a leakage-safe feature set.
4. Add class-specific performance analysis.
5. Add additional model comparison where appropriate.
6. Improve Power BI documentation with dashboard screenshots.
7. Add model monitoring if the solution is later deployed.
8. Validate the workflow on newer healthcare data.

---

# 📌 Final Project Outcome

This project demonstrates an end-to-end healthcare analytics workflow:

```text
Healthcare Data
      ↓
Python Data Analysis
      ↓
Data Cleaning
      ↓
EDA
      ↓
Feature Engineering
      ↓
Machine Learning
      ↓
Model Evaluation
      ↓
Feature Importance
      ↓
Prediction Output
      ↓
Power BI
      ↓
Healthcare Business Insights
```

The project combines **data analytics, machine learning, and business intelligence** to analyze healthcare provider utilization and Medicare payment patterns.

---

## 👩‍💻 Author

**Ankitha G**

Data Analytics & Data Science with Gen AI

GitHub:
https://github.com/ankithag235

LinkedIn:
https://linkedin.com/in/g-ankitha-222838279

# Healthcare Provider Performance, Medicare Cost & Utilization Analytics with Machine Learning

## VS Code project flow — matching the handwritten format

```text
DATA
  ↓
SELECT / LOAD DATA
  ↓
RAW DATA
  ↓
PREPROCESSING
  ↓
PROBLEM STATEMENT
  ↓
EDA
  ↓
INFERENCE
  ↓
FEATURE ENGINEERING
  ↓
MODEL BUILDING
  ↓
COMPARISON OF MODELS
  ↓
FINAL MODEL
  ↓
CROSS VALIDATION
  ↓
FINE-TUNING
  ↓
BEST MODEL
  ↓
PREDICTION
  ↓
TEST EVALUATION
  ↓
INFERENCE
  ↓
TRAINED MODEL EXPORT
  ↓
MODEL PREPARATION / DEPLOYMENT READY
```

## Project objective

Classify healthcare providers into:

- Low Utilization
- Medium Utilization
- High Utilization

using provider, location, service, utilization and financial features.

## Dataset

The project brief specifies:

- 100,000 records
- 27 columns
- U.S. healthcare provider / Medicare data

Put the actual CSV at:

`data/raw/healthcare_providers_capstone1.csv`

The project does not invent dataset results. All metrics, rankings and findings are generated only after the real CSV is supplied.

## Run in VS Code

### 1. Create environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install packages

```bash
pip install -r requirements.txt
```

### 3. Put the dataset in

```text
data/raw/healthcare_providers_capstone1.csv
```

### 4. Run the complete pipeline

```bash
python src/main.py
```

### 5. Run individual stages

```bash
python src/01_load_data.py
python src/02_preprocess.py
python src/03_eda.py
python src/04_feature_engineering.py
python src/05_model_building.py
python src/06_model_comparison.py
python src/07_cross_validation_tuning.py
python src/08_final_model.py
python src/09_prediction.py
python src/10_test_evaluation.py
python src/11_inference.py
python src/12_export_model.py
```

## Outputs

- `data/processed/cleaned_data.csv`
- `data/processed/features.csv`
- `reports/tables/`
- `reports/figures/`
- `models/`
- model metrics and predictions

## Important

The default utilization target is created from service-volume tertiles. This is a documented methodology choice, not an observed dataset result. If your trainer/capstone requires a different definition of Low/Medium/High utilization, change the target logic in `04_feature_engineering.py`.

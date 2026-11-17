# SEC3044 Group Assignment Member 3

## Structure
notebooks/  01_models (CIC models), 02_cross_dataset, 03_shap, 04_unsw_models, 05_report_tables
src/        common.py (load and evaluate helpers)
results/    metrics tables (CSV)   figures/  plots   exports/  trained model

## Setup
python -m venv venv; venv\Scripts\activate; pip install -r requirements.txt

## Data
Not in the repo. Download from xioshi in the whatsapp group and unzip into data/CICIDS2017_output and data/UNSW_output.

## Run order
1) 01_models  2) 04_unsw_models  3) 02_cross_dataset  4) 03_shap  5) 05_report_tables

## Using the model
exports/final_model.joblib: raw features in the order of bundle['features'];
the bundle includes the scaler; attack if probability >= 0.5.

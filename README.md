# CER appointment intelligence

Two trained systems and a local Streamlit application for the supplied rehabilitation appointment extract. See [the measured findings](artifacts/REPORT.md); accuracy targets are assessed honestly and are not all met by this dataset.

## Open the application

In this project folder:

```powershell
python -m pip install -r requirements-lock.txt
python -m streamlit run app.py
```

Trained artifacts are already in `artifacts/`. No retraining is needed to explore the included data. To regenerate every model, comparison and report:

```powershell
python train_models.py
python -m pytest tests -q
```

Use Python 3.13 with `requirements-lock.txt` for the saved artifacts. `requirements.txt` lists broader compatible development versions and requires retraining if versions differ. The training versions are also recorded in `artifacts/metrics.json`. Saved scikit-learn models should be loaded using the same version as training; retrain after dependency changes. Only load model files produced by this project.

## Application workspaces

- **Clinic overview:** attendance rates, daily volume, specialty/location/health segmentation, season/weather exploration and quality notes.
- **No-show risk:** a validated form, calibrated probability and reminder threshold for a single appointment. No SMS is sent; contact-system integration requires a configured provider and clinic-approved workflow.
- **Demand and staffing:** overall/specialty forecasts, optional city-filtered model evaluation, end-date selection, forecast downloads and approximate planning bands.
- **Evidence and impact:** model comparisons, final test metrics, confusion matrix, validation permutation importance, calibration, backtests and assumption-based outreach estimates.

## Data contract

Default input: `Medical_appointment_data.csv`. Keep this source file unchanged. `prepare_data` accepts either readable categories or one-hot columns such as `specialty_physiotherapy`. Dates use `appointment_date_continuous`; outcomes use `no_show` (`yes/no` or `1/0`). The companion field is **`patient_needs_companion`**. Missing and invalid outcomes are rejected. Numeric missing values are imputed inside the fitted pipeline using training dates only; categorical missingness remains `Unknown`. Invalid ages/hours and binary flags are rejected. Unknown cities use a bounded, infrequent-category encoding.

The existing notebook now reads the original extract, reconstructs readable categories and writes derived outputs under `artifacts/`, avoiding repeated overwrites of its source file. It preserves missingness for data-quality analysis; model imputation happens only inside the fitted training pipeline. Missing rainfall is distinct from observed zero rainfall.

## Models and evaluation

Classification compares balanced logistic regression, balanced random forest, histogram gradient boosting, XGBoost and LightGBM. The earliest 60% of dates fit candidates, the next 10% calibrate probabilities, the next 10% select the model by F1 with ROC-AUC as the tie-breaker and select its threshold by F1. The last 20% of dates are held out. Each partition retains natural outcome proportions. The pipeline excludes identifiers, outcomes, SMS receipt and observed same-day weather. Prior-day rain/storm inputs support next-day reminder planning. There is no verified patient ID for grouping repeated patients.

Forecasting compares a weekly seasonal baseline, ridge regression, random forest, XGBoost and LightGBM with lagged volumes, trailing means, weekday, season and trend. Three expanding 28-day validation blocks select by MAE, followed by two 28-day test blocks. Forecasts are recursive within each block. Final operational forecasters are refitted on full history; classifier artifacts retain the evaluated fitted/calibrated model. MAPE excludes zero-volume days and reports their count; MAE, RMSE, WAPE and R² include them. Specialty models are fitted independently and do not necessarily sum to the overall forecast. Forecast bands are validation-error planning ranges, not calibrated coverage guarantees.

Model choice prioritises reproducible tabular methods appropriate to 498 daily observations. XGBoost and LightGBM are included as tuned tree-boosting candidates; recurrent neural networks are not used because this series has only 498 daily observations. Available weather measurements are realised observations, not archived forecasts. They are explored descriptively; demand models use calendar and past demand until prediction-time weather forecasts are available.

## Interpretation limits

The file has 26,289 location labels rather than 13 validated cities, including apparent generated names and damaged encoding. Location strongly influences classification, so model performance needs validation against original records. The date range is January 2020 to May 2021 and has highly irregular daily counts. Booked volume does not measure unmet demand. Verify that the continuous date is an actual service date and that missing calendar days mean zero demand.

The model uses the raw extract and fits preprocessing only on training dates. Accuracy targets and deployment readiness remain unmet where shown in the report. Outreach reduction and staffing figures in the app are explicit scenarios, not measured gains.

## Deployment

Local launch uses the command above and Streamlit's default localhost port 8501. For a hosted deployment, provision an approved private environment, install the pinned training-compatible dependencies, copy the app plus trusted model artifacts and input data, and run the same entrypoint. Add authentication and access controls, transport encryption, log/retention policies, data-version checks, model monitoring and a validated retraining schedule before clinical operation. Do not upload patient data to a public hosting service. No cloud deployment or external messaging has been performed.

## Files

- `clinic_ml.py`: schema handling, feature engineering, evaluation and prediction.
- `train_models.py`: training, persistence, aggregate exports and report generation.
- `app.py`, `app_pages/`, `ui_helpers.py`: application and bounded caches.
- `artifacts/metrics.json`, `artifacts/REPORT.md`: reproducible results and interpretation.
- `tests/`: input, forecasting and UI interaction checks.

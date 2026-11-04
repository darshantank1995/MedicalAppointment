# CER appointment analytics: measured results

Input: 96,337 appointments; observed no-show rate 32.49%.
Date range: 2020-01-01 to 2021-05-12.

## No-show classification

Selected model: **XGBoost**, chosen by validation macro F1 with ROC-AUC as the tie-breaker, using a validation-tuned macro-F1 threshold.
Untouched stratified test: macro F1 **0.683**, balanced accuracy **0.697**, ROC-AUC **0.759**, precision **0.538**, recall **0.670**.
PR-AUC (average precision): 0.560; Brier score: 0.179.
Threshold: 0.465; test prevalence: 32.49%; test appointments: 19,268.
ROC-AUC target attainment (> 0.75): True.
Warm single-appointment prediction p95: 0.149 seconds (10 measurements on this computer).

Confusion matrix rows are actual [attended, missed]; columns are predicted [attended, missed]:
`[[9406, 3601], [2064, 4197]]`

## Overall demand

Selected model: **Weekly seasonal baseline**, chosen by MAE on three expanding 28-day validation blocks.
Final test comprises two successive 28-day forecasts, 2021-03-18 to 2021-05-12.
MAE: **146.73** appointments/day; RMSE: **278.78**; MAPE: **3272.40%**; R²: **-0.078**.
Target attainment: MAPE < 20% = False; R² > 0.65 = False.
28-day prediction latency: 0.001 seconds.
Zero-demand days are excluded only from MAPE; MAE, RMSE, WAPE and R² retain them.

## Specialty forecasts

| Specialty | Model | MAE | MAPE | R² |
|---|---|---:|---:|---:|
| All specialties | Weekly seasonal baseline | 146.73 | 3272.4% | -0.078 |
| assist | Ridge regression | 1.07 | 44.7% | -0.111 |
| enf | Ridge regression | 2.50 | 98.5% | -0.064 |
| general | Weekly seasonal baseline | 26.89 | 414.8% | -0.005 |
| occupational therapy | Weekly seasonal baseline | 16.23 | 144.3% | -0.207 |
| pedagogo | Weekly seasonal baseline | 6.27 | 123.2% | -0.596 |
| physiotherapy | Ridge regression | 35.85 | 1473.4% | -0.006 |
| psychotherapy | Weekly seasonal baseline | 39.75 | 986.4% | -0.019 |
| sem especialidade | Weekly seasonal baseline | 0.73 | 87.6% | -0.132 |
| speech therapy | Ridge regression | 45.62 | 1837.5% | -0.023 |

## Interpretation and operational recommendations

Largest validation permutation signals (decrease in ROC-AUC when shuffled):
- place: 0.1883 (repeat SD 0.0101).
- specialty: 0.0117 (repeat SD 0.0032).
- appointment_hour: 0.0089 (repeat SD 0.0039).
- age: 0.0089 (repeat SD 0.0030).
- disability: 0.0078 (repeat SD 0.0014).

These are predictive associations, not causal explanations. Very small changes imply weak ranking evidence.

1. Use a capacity-limited reminder pilot. Review precision and recall before operational targeting; never cancel or restrict care from a risk score.
2. Compare forecasts with the weekly baseline and retain human schedule review when targets are unmet. Recent demand changes and low daily counts can make percentage errors very large.
3. Audit city labels before geographic resource allocation. This extract has far more than 13 labels, including names that appear generated and text with encoding damage; no automatic mapping to real cities is assumed.
4. Use prior-day rain/storm information for next-day reminders. Same-day observed temperature and rain support descriptive analysis only; operational forecasts need archived weather forecasts available at the prediction time.
5. Collect verified patient identifiers, booking timestamps, cancellations, clinic closures and capacity limits. These enable honest attendance-history features and improve the definition of demand.

## Scenario estimates (not measured intervention effects)

Contacting the highest-risk 20% captured 36.6% of historical test no-shows. If outreach prevents 20% of missed visits in that group, the scenario gives a 7.3% relative reduction, or 2.38 percentage points of all appointments. Effectiveness must be established by a randomised pilot.
Overall test MAE changed by 0.0% relative to a weekly seasonal baseline (positive means improvement). This is a forecast-error comparison, not an observed staffing-efficiency gain. Staff hours require specialty-specific service duration, working hours and utilisation assumptions.

## Data and evaluation limitations

The supplied source extract contains missing values. Numeric imputation is fitted on training dates only; categorical missingness is retained as an Unknown category. This prevents validation and test dates from influencing preprocessing.
No verified patient identifier is present; repeated-patient generalisation cannot be separately measured. Dates are treated as actual service dates and unrecorded calendar dates as zero appointments; verify both assumptions with clinic staff.
Observed record counts are booked-appointment volume, not unmet demand or actual attendance. Do not infer revenue without a verified payment model.
Forecast bands use the 90th percentile of validation absolute errors. They are approximate planning ranges, not guaranteed confidence intervals. Horizons beyond 28 days are extrapolations beyond the evaluated horizon.
The classifier keeps its evaluated fit/calibration split. Forecast models are refitted on all available history only after evaluation.

## Reproducibility and deployment

See metrics.json for data SHA-256, package versions, split dates, comparisons and all metrics. Run `python train_models.py` to regenerate artifacts and `python -m streamlit run app.py` to open the local application.
The local app is a prototype. Hosted deployment requires authentication, appropriate access controls, retention rules, drift monitoring and updated validation. No SMS is sent and no patient contact data is exported by this application.
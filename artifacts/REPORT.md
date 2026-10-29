# CER appointment analytics: measured results

Input: 109,593 appointments; observed no-show rate 31.78%.
Date range: 2020-01-01 to 2021-05-12.

## No-show classification

Selected model: **Histogram gradient boosting**, chosen by validation F1 with ROC-AUC as the tie-breaker, using a validation-tuned F1 threshold.
Untouched chronological test: F1 **0.612**, ROC-AUC **0.761**, precision **0.510**, recall **0.766**.
PR-AUC (average precision): 0.543; Brier score: 0.175.
Threshold: 0.363; test prevalence: 31.09%; test appointments: 20,672.
Target attainment: F1 > 0.70 = False; ROC-AUC > 0.75 = True.
Warm single-appointment prediction p95: 0.208 seconds (10 measurements on this computer).

Confusion matrix rows are actual [attended, missed]; columns are predicted [attended, missed]:
`[[9511, 4734], [1507, 4920]]`

## Overall demand

Selected model: **Weekly seasonal baseline**, chosen by MAE on three expanding 28-day validation blocks.
Final test comprises two successive 28-day forecasts, 2021-03-18 to 2021-05-12.
MAE: **164.98** appointments/day; RMSE: **313.59**; MAPE: **3757.17%**; R²: **-0.077**.
Target attainment: MAPE < 20% = False; R² > 0.65 = False.
28-day prediction latency: 0.002 seconds.
Zero-demand days are excluded only from MAPE; MAE, RMSE, WAPE and R² retain them.

## Specialty forecasts

| Specialty | Model | MAE | MAPE | R² |
|---|---|---:|---:|---:|
| All specialties | Weekly seasonal baseline | 164.98 | 3757.2% | -0.077 |
| assist | Ridge regression | 1.26 | 43.0% | -0.116 |
| enf | Ridge regression | 4.02 | 113.7% | -0.030 |
| general | Weekly seasonal baseline | 29.52 | 424.5% | -0.019 |
| occupational therapy | Weekly seasonal baseline | 18.00 | 473.6% | -0.209 |
| pedagogo | Weekly seasonal baseline | 6.36 | 122.2% | -0.515 |
| physiotherapy | LightGBM | 55.35 | 2258.4% | -0.243 |
| psychotherapy | Weekly seasonal baseline | 42.86 | 951.9% | -0.021 |
| sem especialidade | Weekly seasonal baseline | 0.73 | 87.6% | -0.132 |
| speech therapy | LightGBM | 58.11 | 2850.3% | -0.283 |

## Interpretation and operational recommendations

Largest validation permutation signals (decrease in ROC-AUC when shuffled):
- place: 0.1937 (repeat SD 0.0077).
- age: 0.0145 (repeat SD 0.0021).
- specialty: 0.0099 (repeat SD 0.0025).
- disability: 0.0069 (repeat SD 0.0011).
- chronic_count: 0.0028 (repeat SD 0.0009).

These are predictive associations, not causal explanations. Very small changes imply weak ranking evidence.

1. Use a capacity-limited reminder pilot. Review precision and recall before operational targeting; never cancel or restrict care from a risk score.
2. Compare forecasts with the weekly baseline and retain human schedule review when targets are unmet. Recent demand changes and low daily counts can make percentage errors very large.
3. Audit city labels before geographic resource allocation. This extract has far more than 13 labels, including names that appear generated and text with encoding damage; no automatic mapping to real cities is assumed.
4. Use prior-day rain/storm information for next-day reminders. Same-day observed temperature and rain support descriptive analysis only; operational forecasts need archived weather forecasts available at the prediction time.
5. Collect verified patient identifiers, booking timestamps, cancellations, clinic closures and capacity limits. These enable honest attendance-history features and improve the definition of demand.

## Scenario estimates (not measured intervention effects)

Contacting the highest-risk 20% captured 37.3% of historical test no-shows. If outreach prevents 20% of missed visits in that group, the scenario gives a 7.5% relative reduction, or 2.32 percentage points of all appointments. Effectiveness must be established by a randomised pilot.
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
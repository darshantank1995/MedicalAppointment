"""Reproduce model artifacts and evaluation reports: python train_models.py."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from pathlib import Path
import argparse
import hashlib
import json
import platform
import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from clinic_ml import DATE, prepare_data, fit_classifier, daily_series, fit_forecaster, forecast_table

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / 'Medical_appointment_data_clean_filled.csv'


def write_report(report, output):
    c = report['classification']
    f = report['demand']['All specialties']
    cm, fm = c['test'], f['test']
    lines = [
        '# CER appointment analytics: measured results', '',
        f"Input: {report['data']['rows']:,} appointments; observed no-show rate {report['data']['no_show_rate']:.2%}.",
        f"Date range: {report['data']['start']} to {report['data']['end']}.", '',
        '## No-show classification', '',
        f"Selected model: **{c['selected_model']}**, chosen by validation F1 with ROC-AUC as the tie-breaker, using a validation-tuned F1 threshold.",
        f"Untouched chronological test: F1 **{cm['f1']:.3f}**, ROC-AUC **{cm['roc_auc']:.3f}**, precision **{cm['precision']:.3f}**, recall **{cm['recall']:.3f}**.",
        f"PR-AUC (average precision): {cm['average_precision']:.3f}; Brier score: {cm['brier_score']:.3f}.",
        f"Threshold: {cm['threshold']:.3f}; test prevalence: {cm['no_show_rate']:.2%}; test appointments: {cm['records']:,}.",
        f"Target attainment: F1 > 0.70 = {cm['f1'] > .7}; ROC-AUC > 0.75 = {cm['roc_auc'] > .75}.",
        f"Warm single-appointment prediction p95: {cm['latency_p95_seconds']:.3f} seconds (10 measurements on this computer).", '',
        'Confusion matrix rows are actual [attended, missed]; columns are predicted [attended, missed]:',
        f"`{cm['confusion_matrix']}`", '',
        '## Overall demand', '',
        f"Selected model: **{f['selected_model']}**, chosen by MAE on three expanding 28-day validation blocks.",
        f"Final test comprises two successive 28-day forecasts, {f['test_start']} to {f['test_end']}.",
        f"MAE: **{fm['mae']:.2f}** appointments/day; RMSE: **{fm['rmse']:.2f}**; MAPE: **{fm['mape']:.2%}**; R²: **{fm['r2']:.3f}**.",
        f"Target attainment: MAPE < 20% = {fm['mape'] < .2}; R² > 0.65 = {fm['r2'] > .65}.",
        f"28-day prediction latency: {fm['latency_28_days_seconds']:.3f} seconds.",
        'Zero-demand days are excluded only from MAPE; MAE, RMSE, WAPE and R² retain them.', '',
        '## Specialty forecasts', '',
        '| Specialty | Model | MAE | MAPE | R² |', '|---|---|---:|---:|---:|'
    ]
    for name, item in report['demand'].items():
        m = item['test']
        mape = f"{m['mape']:.1%}" if m['mape'] is not None else 'N/A'
        r2 = f"{m['r2']:.3f}" if m['r2'] is not None else 'N/A'
        lines.append(f"| {name} | {item['selected_model']} | {m['mae']:.2f} | {mape} | {r2} |")
    importance = pd.read_csv(output / 'feature_importance.csv')
    lines += ['', '## Interpretation and operational recommendations', '',
        'Largest validation permutation signals (decrease in ROC-AUC when shuffled):']
    lines += [f"- {row.feature}: {row.auc_decrease:.4f} (repeat SD {row.std:.4f})." for row in importance.head(5).itertuples()]
    lines += ['', 'These are predictive associations, not causal explanations. Very small changes imply weak ranking evidence.', '',
        '1. Use a capacity-limited reminder pilot. Review precision and recall before operational targeting; never cancel or restrict care from a risk score.',
        '2. Compare forecasts with the weekly baseline and retain human schedule review when targets are unmet. Recent demand changes and low daily counts can make percentage errors very large.',
        '3. Audit city labels before geographic resource allocation. This extract has far more than 13 labels, including names that appear generated and text with encoding damage; no automatic mapping to real cities is assumed.',
        '4. Use prior-day rain/storm information for next-day reminders. Same-day observed temperature and rain support descriptive analysis only; operational forecasts need archived weather forecasts available at the prediction time.',
        '5. Collect verified patient identifiers, booking timestamps, cancellations, clinic closures and capacity limits. These enable honest attendance-history features and improve the definition of demand.', '',
        '## Scenario estimates (not measured intervention effects)', '']
    row = next(r for r in c['outreach'] if r['capacity_fraction'] == .2)
    relative = row['share_of_all_no_shows'] * .2
    absolute = cm['no_show_rate'] * relative
    gain = 1 - fm['mae'] / f['baseline_test']['mae'] if f['baseline_test']['mae'] else 0
    lines += [
        f"Contacting the highest-risk 20% captured {row['share_of_all_no_shows']:.1%} of historical test no-shows. If outreach prevents 20% of missed visits in that group, the scenario gives a {relative:.1%} relative reduction, or {absolute * 100:.2f} percentage points of all appointments. Effectiveness must be established by a randomised pilot.",
        f"Overall test MAE changed by {gain:.1%} relative to a weekly seasonal baseline (positive means improvement). This is a forecast-error comparison, not an observed staffing-efficiency gain. Staff hours require specialty-specific service duration, working hours and utilisation assumptions.", '',
        '## Data and evaluation limitations', '',
        'The input was already filled and encoded. Earlier global imputation cannot be undone without the raw extract, so this is exploratory evidence rather than a production validation. Further pipeline imputation is fitted only on training data.',
        'No verified patient identifier is present; repeated-patient generalisation cannot be separately measured. Dates are treated as actual service dates and unrecorded calendar dates as zero appointments; verify both assumptions with clinic staff.',
        'Observed record counts are booked-appointment volume, not unmet demand or actual attendance. Do not infer revenue without a verified payment model.',
        'Forecast bands use the 90th percentile of validation absolute errors. They are approximate planning ranges, not guaranteed confidence intervals. Horizons beyond 28 days are extrapolations beyond the evaluated horizon.',
        'The classifier keeps its evaluated fit/calibration split. Forecast models are refitted on all available history only after evaluation.', '',
        '## Reproducibility and deployment', '',
        'See metrics.json for data SHA-256, package versions, split dates, comparisons and all metrics. Run `python train_models.py` to regenerate artifacts and `python -m streamlit run app.py` to open the local application.',
        'The local app is a prototype. Hosted deployment requires authentication, appropriate access controls, retention rules, drift monitoring and updated validation. No SMS is sent and no patient contact data is exported by this application.'
    ]
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.data)
    d = prepare_data(raw)
    report = {'data': {'file': args.data.name, 'sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'rows': len(d), 'start': str(d[DATE].min().date()), 'end': str(d[DATE].max().date()),
        'no_show_rate': float(d.no_show.mean()), 'location_labels': int(d.place.nunique()),
        'duplicates': int(raw.duplicated().sum()), 'input_missing_cells': int(raw.isna().sum().sum())},
        'versions': {'python': platform.python_version(), 'pandas': pd.__version__,
                     'numpy': np.__version__, 'sklearn': sklearn.__version__,
                     'xgboost': xgboost.__version__, 'lightgbm': lightgbm.__version__}, 'demand': {}, 'skipped_forecasts': {}}
    classifier, report['classification'], importance, reliability = fit_classifier(d)
    joblib.dump(classifier, args.output / 'classifier.joblib', compress=3)
    importance.to_csv(args.output / 'feature_importance.csv', index=False)
    reliability.to_csv(args.output / 'calibration.csv', index=False)
    bundles = {}
    for name in ['All specialties'] + sorted(d.specialty.unique().tolist()):
        print('Forecast:', name, flush=True)
        series = daily_series(d, None if name == 'All specialties' else name)
        try:
            bundles[name], report['demand'][name], backtest = fit_forecaster(series)
        except ValueError as error:
            report['skipped_forecasts'][name] = str(error)
            continue
        slug = ''.join(c if c.isalnum() else '_' for c in name)
        backtest.to_csv(args.output / f'backtest_{slug}.csv', index_label='date')
        forecast_table(bundles[name], 28).to_csv(args.output / f'forecast_{slug}.csv', index=False)
    joblib.dump(bundles, args.output / 'forecasters.joblib', compress=3)
    for field in ['specialty', 'place', 'gender', 'disability', 'appointment_shift',
                  'hipertension', 'diabetes', 'patient_needs_companion', 'day_of_week']:
        summary = d.groupby(field, dropna=False).agg(appointments=('no_show', 'size'), no_show_rate=('no_show', 'mean')).sort_values('appointments', ascending=False)
        summary.head(30).to_csv(args.output / f'eda_{field}.csv')
    daily_series(d).to_csv(args.output / 'daily_demand.csv', index_label='date')
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    write_report(report, args.output)
    print('Saved trained models, metrics and report to', args.output, flush=True)


if __name__ == '__main__':
    main()

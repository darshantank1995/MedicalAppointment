"""Data preparation and time-aware evaluation for CER appointment planning.

No training operation modifies the supplied appointment extract. All preprocessing
is fitted on training rows. Outcome 1 means a missed appointment.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score,
    r2_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

SEED = 42
DATE = 'appointment_date_continuous'
CATEGORIES = ['gender', 'specialty', 'disability', 'appointment_shift']
NUMERIC = ['age', 'appointment_hour', 'patient_needs_companion', 'hipertension',
    'diabetes', 'alcoholism', 'handcap', 'scholarship', 'rainy_day_before',
    'storm_day_before', 'under_12', 'over_60', 'day_of_week', 'month_sin',
    'month_cos', 'day_of_year_sin', 'day_of_year_cos', 'companion_child', 'chronic_count']
CATEGORICAL = CATEGORIES + ['place']
FEATURES = NUMERIC + CATEGORICAL
BINARY = ['patient_needs_companion', 'hipertension', 'diabetes', 'alcoholism',
    'handcap', 'scholarship', 'rainy_day_before', 'storm_day_before']


def target_values(series):
    """Decode outcome labels explicitly; never silently turn unknown outcomes into 0."""
    mapping = {'yes': 1, '1': 1, '1.0': 1, 'true': 1, 'missed': 1,
               'no': 0, '0': 0, '0.0': 0, 'false': 0, 'attended': 0}
    result = series.astype('string').str.strip().str.lower().map(mapping)
    if result.isna().any():
        raise ValueError('Missing or unrecognised no_show labels. Expected yes/no or 1/0.')
    return result.astype(int)


def prepare_data(raw):
    """Read either raw categorical columns or the existing one-hot encoded extract."""
    d = raw.copy()
    d.columns = d.columns.astype(str).str.strip().str.lower().str.replace('-', '_', regex=False)
    if d.columns.duplicated().any():
        raise ValueError('Duplicate column names after header normalisation.')
    for category in CATEGORIES:
        if category not in d:
            cols = [c for c in d if c.startswith(category + '_')]
            if cols:
                indicators = d[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
                valid = indicators.eq(1).sum(axis=1).eq(1) & indicators.isin([0, 1]).all(axis=1)
                labels = indicators.idxmax(axis=1).str[len(category) + 1:]
                d[category] = labels.where(valid, 'Unknown')
            else:
                d[category] = 'Unknown'
        d[category] = d[category].astype('string').str.strip().replace({'nan': 'Unknown', '': 'Unknown'}).fillna('Unknown')
    if DATE not in d:
        raise ValueError(f'Missing required date column: {DATE}')
    d[DATE] = pd.to_datetime(d[DATE], errors='coerce').dt.normalize()
    if d[DATE].isna().any():
        raise ValueError('Appointment dates contain missing or invalid values.')
    if 'no_show' in d:
        d['no_show'] = target_values(d['no_show'])
    if 'patient_needs_companion' not in d and 'needs_companion' in d:
        d['patient_needs_companion'] = d['needs_companion']
    if 'appointment_hour' not in d and 'appointment_time' in d:
        d['appointment_hour'] = pd.to_numeric(d['appointment_time'], errors='coerce')
    for c in NUMERIC:
        if c not in d:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if ((d.age < 0) | (d.age > 120)).any():
        raise ValueError('Age must be between 0 and 120, or missing.')
    if ((d.appointment_hour < 0) | (d.appointment_hour > 23)).any():
        raise ValueError('Appointment hour must be between 0 and 23.')
    for c in BINARY:
        if not d[c].dropna().isin([0, 1]).all():
            raise ValueError(f'{c} must contain 0, 1, or missing values.')
    if 'place' not in d:
        d['place'] = 'Unknown'
    d['place'] = d.place.astype('string').str.strip().fillna('Unknown')
    d['under_12'] = (d.age < 12).astype(float).where(d.age.notna())
    d['over_60'] = (d.age > 60).astype(float).where(d.age.notna())
    d['day_of_week'] = d[DATE].dt.dayofweek
    d['month_sin'] = np.sin(2 * np.pi * d[DATE].dt.month / 12)
    d['month_cos'] = np.cos(2 * np.pi * d[DATE].dt.month / 12)
    d['day_of_year_sin'] = np.sin(2 * np.pi * d[DATE].dt.dayofyear / 365.25)
    d['day_of_year_cos'] = np.cos(2 * np.pi * d[DATE].dt.dayofyear / 365.25)
    d['companion_child'] = d.patient_needs_companion * d.under_12
    d['chronic_count'] = d[['hipertension', 'diabetes']].sum(axis=1, min_count=2)
    return d.sort_values(DATE, kind='stable').reset_index(drop=True)


def load_analysis_data(path):
    """Readable notebook frame; reconstruct encoded categories without duplicate indicators."""
    d = prepare_data(pd.read_csv(path))
    prefixes = tuple(name + '_' for name in CATEGORIES + ['age_group', 'appointment_time_band',
                                                         'appointment_day_of_week', 'rain_intensity', 'heat_intensity'])
    d = d.drop(columns=[c for c in d if c.startswith(prefixes)])
    d['no_show'] = d.no_show.map({0: 'no', 1: 'yes'})
    return d


def no_show_summary(data, column, top_n=12):
    """Vectorised summary supporting numeric and text outcomes without per-group decoding."""
    if column == 'needs_companion' and column not in data:
        column = 'patient_needs_companion'
    if column not in data:
        raise ValueError(f'Missing grouping column: {column}')
    result = (data.assign(_no_show_flag=target_values(data['no_show']))
        .groupby(column, dropna=False)
        .agg(total=('no_show', 'size'), no_show_rate=('_no_show_flag', 'mean'))
        .sort_values('total', ascending=False).head(top_n).reset_index())
    result['no_show_rate'] = (result['no_show_rate'] * 100).round(2)
    return result


def make_preprocessor():
    return ColumnTransformer([
        ('numbers', Pipeline([('missing', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                              ('scale', StandardScaler())]), NUMERIC),
        ('categories', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                     min_frequency=100, max_categories=20, sparse_output=False), CATEGORICAL)
    ])


def classification_metrics(y, probabilities, threshold):
    predictions = probabilities >= threshold
    return {'f1': float(f1_score(y, predictions, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, probabilities)),
        'average_precision': float(average_precision_score(y, probabilities)),
        'precision': float(precision_score(y, predictions, zero_division=0)),
        'recall': float(recall_score(y, predictions, zero_division=0)),
        'brier_score': float(brier_score_loss(y, probabilities)),
        'threshold': float(threshold), 'no_show_rate': float(np.mean(y)),
        'records': int(len(y)), 'confusion_matrix': confusion_matrix(y, predictions, labels=[0, 1]).tolist()}


def fit_classifier(d):
    """60/10/10/20% date blocks: fit, calibrate, select, then untouched test."""
    dates = d[DATE].drop_duplicates().sort_values().to_numpy()
    if len(dates) < 100 or 'no_show' not in d:
        raise ValueError('Classification requires labelled data spanning at least 100 dates.')
    cuts = dates[[int(len(dates) * p) for p in [0.6, 0.7, 0.8]]]
    fit = d[d[DATE] < cuts[0]]
    calibrate = d[(d[DATE] >= cuts[0]) & (d[DATE] < cuts[1])]
    validation = d[(d[DATE] >= cuts[1]) & (d[DATE] < cuts[2])]
    test = d[d[DATE] >= cuts[2]]
    for frame in [fit, calibrate, validation, test]:
        if frame.no_show.nunique() != 2:
            raise ValueError('Every chronological classification partition must contain both outcomes.')
    estimators = {
        'Logistic regression': LogisticRegression(max_iter=600, class_weight='balanced', random_state=SEED),
        'Random forest': RandomForestClassifier(n_estimators=140, max_depth=12,
            min_samples_leaf=20, class_weight='balanced_subsample', n_jobs=2, random_state=SEED),
        'Histogram gradient boosting': HistGradientBoostingClassifier(max_iter=160,
            max_leaf_nodes=15, l2_regularization=10, class_weight='balanced', random_state=SEED),
        'XGBoost': XGBClassifier(n_estimators=220, max_depth=5, learning_rate=.05,
            min_child_weight=10, subsample=.8, colsample_bytree=.8, reg_lambda=10,
            eval_metric='logloss', n_jobs=2, random_state=SEED),
        'LightGBM': LGBMClassifier(n_estimators=220, num_leaves=15, learning_rate=.05,
            min_child_samples=30, subsample=.8, colsample_bytree=.8, reg_lambda=10,
            class_weight='balanced', n_jobs=2, random_state=SEED, verbosity=-1)
    }
    candidates, rows = {}, []
    for name, estimator in estimators.items():
        print('Classification:', name, flush=True)
        pipeline = Pipeline([('prepare', make_preprocessor()), ('estimate', estimator)])
        pipeline.fit(fit[FEATURES], fit.no_show)
        model = CalibratedClassifierCV(FrozenEstimator(pipeline), method='sigmoid')
        model.fit(calibrate[FEATURES], calibrate.no_show)
        p = model.predict_proba(validation[FEATURES])[:, 1]
        thresholds = np.unique(np.r_[np.linspace(.05, .95, 91), np.quantile(p, np.linspace(0, 1, 101))])
        threshold = max(thresholds, key=lambda t: f1_score(validation.no_show, p >= t, zero_division=0))
        metrics = classification_metrics(validation.no_show, p, threshold)
        rows.append({'model': name, **{k: v for k, v in metrics.items() if k != 'confusion_matrix'}})
        candidates[name] = (model, threshold)
    # The stated clinic target requires both F1 and ROC-AUC. Select first by
    # F1, then use ROC-AUC as the tie-breaker; the final test remains untouched.
    scores = pd.DataFrame(rows).sort_values(['f1', 'roc_auc'], ascending=False)
    selected = scores.iloc[0]['model']
    model, threshold = candidates[selected]
    p = model.predict_proba(test[FEATURES])[:, 1]
    test_metrics = classification_metrics(test.no_show, p, threshold)
    sample = validation.sample(min(2000, len(validation)), random_state=SEED)
    importance = permutation_importance(model, sample[FEATURES], sample.no_show,
        scoring='roc_auc', n_repeats=3, random_state=SEED, n_jobs=1)
    importance = pd.DataFrame({'feature': FEATURES, 'auc_decrease': importance.importances_mean,
        'std': importance.importances_std}).sort_values('auc_decrease', ascending=False)
    model.predict_proba(test[FEATURES].iloc[:1])
    timings = []
    for _ in range(10):
        start = perf_counter()
        model.predict_proba(test[FEATURES].iloc[:1])
        timings.append(perf_counter() - start)
    test_metrics['latency_p95_seconds'] = float(np.quantile(timings, .95))
    bins = pd.cut(p, np.linspace(0, 1, 11), include_lowest=True)
    reliability = pd.DataFrame({'bin': bins, 'predicted': p, 'observed': test.no_show.to_numpy()}).groupby('bin', observed=True).agg(
        predicted=('predicted', 'mean'), observed=('observed', 'mean'), count=('observed', 'size')).reset_index(drop=True)
    ranked = np.argsort(-p, kind='stable')
    outreach = []
    for share in [.05, .1, .2, .3]:
        n = max(1, int(len(p) * share))
        missed = int(test.no_show.to_numpy()[ranked[:n]].sum())
        outreach.append({'capacity_fraction': share, 'appointments_contacted': n,
            'historical_no_shows_in_group': missed, 'group_no_show_rate': missed / n,
            'share_of_all_no_shows': missed / int(test.no_show.sum())})
    bundle = {'model': model, 'threshold': float(threshold), 'selected_model': selected,
        'defaults': {c: float(fit[c].median()) if fit[c].notna().any() else 0.0 for c in NUMERIC},
        'options': {c: fit[c].value_counts().head(20).index.tolist() for c in CATEGORICAL}}
    report = {'selected_model': selected, 'test': test_metrics,
        'validation_comparison': scores.to_dict('records'), 'outreach': outreach,
        'splits': {name: {'start': str(frame[DATE].min().date()), 'end': str(frame[DATE].max().date()),
                          'rows': len(frame)} for name, frame in zip(['fit', 'calibration', 'validation', 'test'], [fit, calibrate, validation, test])},
        'targets_met': {'f1_above_0_70': test_metrics['f1'] > .7, 'roc_auc_above_0_75': test_metrics['roc_auc'] > .75}}
    return bundle, report, importance, reliability


def daily_series(d, specialty=None, place=None):
    filtered = d
    if specialty is not None:
        filtered = filtered[filtered.specialty == specialty]
    if place is not None:
        filtered = filtered[filtered.place == place]
    index = pd.date_range(d[DATE].min(), d[DATE].max(), freq='D')
    return filtered.groupby(DATE).size().reindex(index, fill_value=0).astype(float).rename('appointments')


def demand_row(date, history, start_date):
    """All lag inputs end strictly before the day being forecast."""
    row = {'day_of_week': date.dayofweek, 'weekend': int(date.dayofweek >= 5),
           'month_sin': np.sin(2 * np.pi * date.month / 12), 'month_cos': np.cos(2 * np.pi * date.month / 12),
           'year_sin': np.sin(2 * np.pi * date.dayofyear / 365.25),
           'year_cos': np.cos(2 * np.pi * date.dayofyear / 365.25),
           'trend': (date - start_date).days}
    for lag in [1, 7, 14, 28]:
        row[f'lag_{lag}'] = float(history[-lag])
    for window in [7, 28]:
        row[f'mean_{window}'] = float(np.mean(history[-window:]))
    return row


def fit_demand_estimator(series, name):
    if name == 'Weekly seasonal baseline':
        return None
    X = pd.DataFrame([demand_row(series.index[i], series.iloc[:i].to_numpy(), series.index[0])
                      for i in range(28, len(series))])
    y = series.iloc[28:].to_numpy()
    if name == 'Ridge regression':
        model = Pipeline([('scale', StandardScaler()), ('model', Ridge(alpha=100))])
    elif name == 'Random forest':
        model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5,
                                      n_jobs=2, random_state=SEED)
    elif name == 'XGBoost':
        model = XGBRegressor(n_estimators=220, max_depth=4, learning_rate=.05,
            min_child_weight=8, subsample=.8, colsample_bytree=.8, reg_lambda=10,
            objective='reg:squarederror', n_jobs=2, random_state=SEED)
    elif name == 'LightGBM':
        model = LGBMRegressor(n_estimators=220, num_leaves=15, learning_rate=.05,
            min_child_samples=15, subsample=.8, colsample_bytree=.8, reg_lambda=10,
            n_jobs=2, random_state=SEED, verbosity=-1)
    else:
        raise ValueError(f'Unknown demand estimator: {name}')
    model.fit(X, y)
    return model


def recursive_forecast(model, name, history, horizon):
    """A fixed-origin forecast: later holdout observations are never used as lags."""
    if not 1 <= horizon <= 90:
        raise ValueError('Forecast horizon must be 1 to 90 days.')
    values = history.tolist()
    dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon)
    predictions = []
    for date in dates:
        if model is None:
            prediction = values[-7]
        else:
            row = pd.DataFrame([demand_row(date, values, history.index[0])])
            prediction = float(model.predict(row)[0])
        prediction = max(0., prediction)
        values.append(prediction)
        predictions.append(prediction)
    return pd.Series(predictions, index=dates, name='forecast')


def regression_metrics(actual, predicted):
    y, p = np.asarray(actual), np.asarray(predicted)
    nonzero = y != 0
    return {'mae': float(mean_absolute_error(y, p)), 'rmse': float(np.sqrt(mean_squared_error(y, p))),
        'mape': float(np.mean(np.abs((y[nonzero] - p[nonzero]) / y[nonzero]))) if nonzero.any() else None,
        'wape': float(np.abs(y - p).sum() / np.abs(y).sum()) if np.abs(y).sum() else None,
        'r2': float(r2_score(y, p)) if np.var(y) > 0 else None,
        'zero_days_excluded_from_mape': int((~nonzero).sum()), 'days': len(y)}


def fit_forecaster(series):
    """Compare baselines and regressors on expanding 28-day windows; test on last 56 days."""
    if len(series) < 200 or (series > 0).sum() < 60:
        raise ValueError('Forecasting needs at least 200 calendar days and 60 days with appointments.')
    if not series.index.equals(pd.date_range(series.index.min(), series.index.max(), freq='D')):
        raise ValueError('Demand history must have a complete daily calendar.')
    names = ['Weekly seasonal baseline', 'Ridge regression', 'Random forest', 'XGBoost', 'LightGBM']
    test_start = len(series) - 56
    origins = [test_start - 84, test_start - 56, test_start - 28]
    rows, residuals = [], {}
    for name in names:
        actual, predicted = [], []
        for origin in origins:
            history = series.iloc[:origin]
            estimator = fit_demand_estimator(history, name)
            p = recursive_forecast(estimator, name, history, 28)
            actual.extend(series.iloc[origin:origin + 28].tolist())
            predicted.extend(p.tolist())
        metrics = regression_metrics(actual, predicted)
        rows.append({'model': name, **metrics})
        residuals[name] = float(np.quantile(np.abs(np.asarray(actual) - np.asarray(predicted)), .9))
    selected = min(rows, key=lambda row: row['mae'])['model']
    predictions, baseline_predictions = [], []
    for origin in [test_start, test_start + 28]:
        history = series.iloc[:origin]
        estimator = fit_demand_estimator(history, selected)
        predictions.extend(recursive_forecast(estimator, selected, history, 28).tolist())
        baseline_predictions.extend(recursive_forecast(None, names[0], history, 28).tolist())
    actual = series.iloc[test_start:]
    metrics = regression_metrics(actual, predictions)
    baseline = regression_metrics(actual, baseline_predictions)
    model = fit_demand_estimator(series, selected)
    bundle = {'model': model, 'name': selected, 'history': series, 'error_band': residuals[selected]}
    start = perf_counter()
    recursive_forecast(model, selected, series, 28)
    metrics['latency_28_days_seconds'] = perf_counter() - start
    report = {'selected_model': selected, 'test': metrics, 'baseline_test': baseline,
        'validation_comparison': rows, 'test_start': str(actual.index[0].date()),
        'test_end': str(actual.index[-1].date()), 'evaluation_horizon_days': 28,
        'targets_met': {'mape_below_0_20': metrics['mape'] is not None and metrics['mape'] < .2,
                        'r2_above_0_65': metrics['r2'] is not None and metrics['r2'] > .65}}
    backtest = pd.DataFrame({'actual': actual, 'forecast': predictions, 'weekly_baseline': baseline_predictions})
    return bundle, report, backtest


def forecast_table(bundle, horizon):
    p = recursive_forecast(bundle['model'], bundle['name'], bundle['history'], horizon)
    return pd.DataFrame({'date': p.index, 'forecast': p.to_numpy(),
        'lower_planning_band': np.maximum(0, p.to_numpy() - bundle['error_band']),
        'upper_planning_band': p.to_numpy() + bundle['error_band']})

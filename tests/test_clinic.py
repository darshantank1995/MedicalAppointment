"""Checks for input correctness, recursive forecasting and saved-model usability."""
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
import pytest
from clinic_ml import (DATE, FEATURES, prepare_data, target_values, recursive_forecast,
    regression_metrics, load_analysis_data, no_show_summary)
from train_models import DEFAULT_DATA
from ui_helpers import DATA

ROOT = Path(__file__).resolve().parents[1]


def test_training_and_dashboard_use_original_source_extract():
    assert DEFAULT_DATA.name == 'Medical_appointment_data.csv'
    assert DATA == DEFAULT_DATA


def test_encoded_categories_and_outcome_are_recovered():
    raw = pd.DataFrame({DATE: ['2021-01-01', '2021-01-02'], 'age': [11, 61],
        'gender_f': [1, 0], 'gender_m': [0, 1], 'specialty_physiotherapy': [1, 1],
        'patient_needs_companion': [1, 0], 'no_show': ['yes', 'no']})
    result = prepare_data(raw)
    assert result.gender.tolist() == ['f', 'm']
    assert result.specialty.tolist() == ['physiotherapy', 'physiotherapy']
    assert result.no_show.tolist() == [1, 0]
    assert result.under_12.tolist() == [1, 0]
    assert result.over_60.tolist() == [0, 1]
    assert raw.columns.tolist() == [DATE, 'age', 'gender_f', 'gender_m', 'specialty_physiotherapy', 'patient_needs_companion', 'no_show']


def test_invalid_outcomes_and_dates_are_rejected():
    with pytest.raises(ValueError, match='labels'):
        target_values(pd.Series(['yes', 'unknown']))
    with pytest.raises(ValueError, match='dates'):
        prepare_data(pd.DataFrame({DATE: ['not a date']}))
    assert target_values(pd.Series([1., 0., True, False])).tolist() == [1, 0, 1, 0]


def test_ambiguous_one_hot_becomes_unknown():
    result = prepare_data(pd.DataFrame({DATE: ['2021-01-01'], 'gender_f': [1], 'gender_m': [1]}))
    assert result.gender.iloc[0] == 'Unknown'


def test_recursive_forecast_feeds_its_own_predictions():
    class NextValue:
        def predict(self, frame):
            return frame['lag_1'].to_numpy() + 1
    history = pd.Series(np.arange(1., 36.), index=pd.date_range('2021-01-01', periods=35))
    original = history.copy()
    p = recursive_forecast(NextValue(), 'test', history, 3)
    assert p.tolist() == [36., 37., 38.]
    pd.testing.assert_series_equal(history, original)
    assert p.index[0] > history.index[-1]


def test_zero_day_metric_policy():
    metrics = regression_metrics([0, 10], [2, 8])
    assert metrics['mae'] == 2
    assert metrics['mape'] == .2
    assert metrics['zero_days_excluded_from_mape'] == 1


def test_companion_alias_and_no_show_rates():
    data = pd.DataFrame({'patient_needs_companion': [1, 1, 0], 'no_show': ['yes', 'no', 'yes']})
    result = no_show_summary(data, 'needs_companion').set_index('patient_needs_companion')
    assert result.loc[1, 'no_show_rate'] == 50
    assert result.loc[0, 'no_show_rate'] == 100


def test_notebook_frame_has_real_companion_column_and_readable_categories():
    data = load_analysis_data(ROOT / 'Medical_appointment_data_clean_filled.csv')
    assert 'patient_needs_companion' in data
    assert set(data.no_show.unique()) == {'yes', 'no'}
    assert 'gender_f' not in data
    assert 'gender' in data and 'specialty' in data


def test_saved_classifier_handles_new_city_and_missing_values():
    bundle = joblib.load(ROOT / 'artifacts' / 'classifier.joblib')
    raw = pd.DataFrame({DATE: ['2021-05-13'], 'age': [np.nan], 'place': ['UNSEEN_CITY'], 'specialty': ['physiotherapy']})
    p = bundle['model'].predict_proba(prepare_data(raw)[FEATURES])
    assert p.shape == (1, 2)
    assert np.isfinite(p).all()
    assert np.isclose(p.sum(), 1)


def test_predictors_exclude_outcomes_and_realised_same_day_weather():
    assert not set(FEATURES) & {'no_show', 'sms_received', 'average_temp_day', 'average_rain_day', 'max_temp_day', 'max_rain_day'}


def test_saved_comparisons_include_boosted_models():
    metrics = json.loads((ROOT / 'artifacts' / 'metrics.json').read_text(encoding='utf-8'))
    assert {'xgboost', 'lightgbm'} <= set(metrics['versions'])
    classification = {row['model'] for row in metrics['classification']['validation_comparison']}
    demand = {row['model'] for row in metrics['demand']['All specialties']['validation_comparison']}
    assert {'XGBoost', 'LightGBM'} <= classification
    assert {'XGBoost', 'LightGBM'} <= demand

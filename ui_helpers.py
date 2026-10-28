"""Bounded caches for local aggregate data and trusted locally trained artifacts."""
from pathlib import Path
import hashlib
import json
import joblib
import pandas as pd
import streamlit as st
from clinic_ml import prepare_data

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'Medical_appointment_data_clean_filled.csv'
ARTIFACTS = ROOT / 'artifacts'


@st.cache_data(max_entries=2, show_spinner='Loading appointment history...')
def load_data(path, modified):
    return prepare_data(pd.read_csv(path)), hashlib.sha256(Path(path).read_bytes()).hexdigest()


@st.cache_resource(max_entries=2, show_spinner='Loading trained models...')
def load_models(modified):
    # Only project artifacts are loaded; never accept uploaded pickle/joblib files.
    report = json.loads((ARTIFACTS / 'metrics.json').read_text(encoding='utf-8'))
    classifier = joblib.load(ARTIFACTS / 'classifier.joblib')
    forecasters = joblib.load(ARTIFACTS / 'forecasters.joblib')
    return report, classifier, forecasters


def context():
    if not DATA.exists() or not (ARTIFACTS / 'metrics.json').exists():
        st.info('Add the appointment CSV and run python train_models.py to prepare the models.')
        st.stop()
    try:
        d, fingerprint = load_data(str(DATA), DATA.stat().st_mtime_ns)
        report, classifier, forecasts = load_models((ARTIFACTS / 'metrics.json').stat().st_mtime_ns)
    except (ValueError, OSError, KeyError) as error:
        st.error(f'Unable to load the project: {error}')
        st.stop()
    if fingerprint != report['data']['sha256']:
        st.error('The appointment file has changed since training. Run python train_models.py before using these models.')
        st.stop()
    return d, report, classifier, forecasts


def metric_row(items):
    with st.container(horizontal=True):
        for label, value in items:
            st.metric(label, value, border=True)


def label(name):
    return name.replace('_', ' ').capitalize()

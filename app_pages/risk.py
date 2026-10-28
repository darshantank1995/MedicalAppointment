from time import perf_counter
import numpy as np
import pandas as pd
import streamlit as st
from clinic_ml import DATE, FEATURES, BINARY, prepare_data
from ui_helpers import context, metric_row, label

st.title('No-show risk')
st.caption('Estimate no-show risk for a single appointment to support reminder review.')
d, report, bundle, _ = context()
metrics = report['classification']['test']
metric_row([('Test ROC-AUC', f"{metrics['roc_auc']:.3f}"), ('Test F1', f"{metrics['f1']:.3f}"),
            ('Reminder threshold', f"{bundle['threshold']:.1%}")])
if not all(report['classification']['targets_met'].values()):
    st.warning('This model does not meet all requested performance targets. Review the evidence before using it to target reminders.')
st.caption('Prediction time: the evening before the appointment, when prior-day weather is known. Scores support outreach; they must not restrict access to care.')

values = {}
with st.form('appointment_form'):
    st.subheader('Appointment details')
    left, right = st.columns(2)
    values[DATE] = left.date_input('Appointment date', value=(d[DATE].max() + pd.Timedelta(days=1)).date())
    values['appointment_hour'] = right.number_input('Appointment hour (0–23)', min_value=0, max_value=23, value=9)
    values['age'] = left.number_input('Patient age', min_value=0.0, max_value=120.0, value=float(bundle['defaults']['age']))
    for i, field in enumerate(['specialty', 'gender', 'disability', 'appointment_shift', 'place']):
        options = bundle['options'][field]
        options = options + (['Unknown'] if 'Unknown' not in options else [])
        values[field] = [left, right][i % 2].selectbox(label(field), options, key=f'risk_{field}')
    st.caption('Location choices are frequent labels from the extract and need clinic validation. Unknown or less frequent values are handled by the model.')
    st.subheader('Patient and prior-day conditions')
    display_names = {'hipertension': 'Hypertension', 'handcap': 'Recorded disability flag',
        'scholarship': 'Scholarship / support flag', 'patient_needs_companion': 'Patient needs a companion',
        'rainy_day_before': 'Rain on the day before', 'storm_day_before': 'Storm on the day before'}
    for i, field in enumerate(BINARY):
        choice = [left, right][i % 2].selectbox(display_names.get(field, label(field)),
            ['Unknown', 'No', 'Yes'], key=f'risk_{field}')
        values[field] = {'Unknown': np.nan, 'No': 0, 'Yes': 1}[choice]
    submitted = st.form_submit_button('Calculate risk', type='primary')
if submitted:
    try:
        row = prepare_data(pd.DataFrame([values]))
        start = perf_counter()
        p = float(bundle['model'].predict_proba(row[FEATURES])[0, 1])
        elapsed = perf_counter() - start
        st.metric('Estimated probability of missing the appointment', f'{p:.1%}')
        st.info('Prioritise for reminder review' if p >= bundle['threshold'] else 'Standard reminder pathway')
        st.caption(f'Prediction completed in {elapsed:.3f} seconds. This is an estimate, not a certainty about this patient.')
    except (ValueError, KeyError) as error:
        st.error(str(error))

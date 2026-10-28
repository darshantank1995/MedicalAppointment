import numpy as np
import pandas as pd
import streamlit as st
from clinic_ml import daily_series, fit_forecaster, forecast_table
from ui_helpers import context, metric_row

st.title('Demand and staffing')
st.caption('Booked-appointment volume, forecast uncertainty and capacity scenarios')
d, report, _, bundles = context()
specialty = st.selectbox('Specialty', list(bundles), key='forecast_specialty')
location = st.selectbox('Location scope', ['All locations'] + d.place.value_counts().head(11).index.tolist(), key='forecast_location')
st.caption('Location labels are unverified. Use filtered results for exploratory review until the city mapping is confirmed.')

@st.cache_resource(max_entries=4, show_spinner='Evaluating this location forecast...')
def location_forecaster(fingerprint, specialty_name, city):
    series = daily_series(d, None if specialty_name == 'All specialties' else specialty_name, city)
    return fit_forecaster(series)

key = (specialty, location, report['data']['sha256'])
if location != 'All locations':
    if st.button('Prepare location forecast'):
        try:
            st.session_state['location_model'] = (key, location_forecaster(report['data']['sha256'], specialty, location))
        except ValueError as error:
            st.error(str(error))
    stored = st.session_state.get('location_model')
    if stored is None or stored[0] != key:
        st.info('Prepare this location forecast to evaluate its history and calculate predictions.')
        st.stop()
    bundle, evidence, backtest = stored[1]
else:
    bundle = bundles[specialty]
    evidence = report['demand'][specialty]

start_date = (bundle['history'].index[-1] + pd.Timedelta(days=1)).date()
end_date = st.date_input('Forecast through', value=start_date + pd.Timedelta(days=27),
    min_value=start_date, max_value=start_date + pd.Timedelta(days=89), key='forecast_end')
horizon = (end_date - start_date).days + 1
st.info(f'Forecast starts {start_date:%d %b %Y}, immediately after the last available observation. Load recent history and retrain for current service planning.')
if horizon > 28:
    st.warning('This horizon exceeds the evaluated 28-day horizon. Treat the longer outlook as exploratory.')
metrics = evidence['test']
metric_row([('Test MAE', f"{metrics['mae']:.1f} visits/day"),
    ('Test MAPE', f"{metrics['mape']:.1%}" if metrics['mape'] is not None else 'Not defined'),
    ('Test R²', f"{metrics['r2']:.3f}" if metrics['r2'] is not None else 'Not defined')])
if not all(evidence['targets_met'].values()):
    st.warning('The forecast does not meet all requested accuracy targets. Use it alongside the appointment book and staff review.')
st.caption(f"Model: {bundle['name']}. Test period: {evidence['test_start']} to {evidence['test_end']}. MAPE excludes zero-demand days.")
forecast = forecast_table(bundle, horizon)
chart = pd.concat([bundle['history'].tail(90).rename('Historical'), forecast.set_index('date')['forecast'].rename('Forecast')], axis=1)
st.line_chart(chart, y_label='Appointments')
st.dataframe(forecast.round({c: 1 for c in forecast if c != 'date'}), hide_index=True)
st.caption('Planning bands are based on validation errors. They are approximate ranges, not guaranteed confidence intervals. Daily counts include zero-volume days and do not model clinic closures separately.')
st.download_button('Download forecast CSV', forecast.to_csv(index=False), 'demand_forecast.csv', 'text/csv')

import pandas as pd
import streamlit as st
from clinic_ml import DATE, daily_series
from ui_helpers import context, metric_row

st.title('Clinic overview')
st.caption('Appointment patterns, service demand and data quality')
d, report, _, _ = context()
metric_row([('Appointments', f'{len(d):,}'), ('No-show rate', f'{d.no_show.mean():.1%}'),
            ('Specialty labels', str(d.specialty.nunique()))])
st.info(f"Historical data: {d[DATE].min():%d %b %Y} to {d[DATE].max():%d %b %Y}. New appointment data is required for current operational forecasts.")
st.warning(f'The file contains {d.place.nunique():,} location labels, including apparent generated names and encoding issues. Validate city mappings before regional allocation.')
with st.container(border=True):
    st.subheader('Daily booked appointments')
    st.line_chart(daily_series(d), y_label='Appointments')

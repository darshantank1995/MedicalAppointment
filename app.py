"""Run locally with: python -m streamlit run app.py."""
import streamlit as st

st.set_page_config(page_title='CER appointment intelligence', page_icon=':material/local_hospital:', layout='wide')
st.sidebar.title('CER appointment intelligence')
st.sidebar.caption('Rehabilitation service planning')
page = st.navigation([
    st.Page('app_pages/overview.py', title='Clinic overview', icon=':material/monitoring:'),
    st.Page('app_pages/risk.py', title='No-show risk', icon=':material/person_check:'),
    st.Page('app_pages/demand.py', title='Demand and staffing', icon=':material/calendar_month:'),
    st.Page('app_pages/evidence.py', title='Evidence and impact', icon=':material/analytics:')
])
st.sidebar.caption('Local prototype. Human review is required before outreach or staffing changes.')
page.run()

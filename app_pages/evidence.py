import pandas as pd
import streamlit as st
from ui_helpers import ARTIFACTS, context, metric_row

st.title('Evidence and potential impact')
d, report, _, _ = context()
c = report['classification']
st.subheader('No-show model comparison')
st.caption('Candidates fitted on the earliest 60% of dates; probabilities calibrated on the next 10%; model selected by F1 with ROC-AUC as the tie-breaker and threshold selected on the next 10%. Final 20% of dates used only for testing. Natural class proportions retained.')
st.dataframe(pd.DataFrame(c['validation_comparison']), hide_index=True)
st.write('Final test metrics', c['test'])
matrix = pd.DataFrame(c['test']['confusion_matrix'], index=['Actually attended', 'Actually missed'], columns=['Predicted attended', 'Predicted missed'])
st.table(matrix)
importance = pd.read_csv(ARTIFACTS / 'feature_importance.csv')
st.bar_chart(importance.head(12).set_index('feature')['auc_decrease'], horizontal=True)
st.caption('Permutation importance measures the change in validation ROC-AUC when a field is shuffled. Negative or near-zero values provide little evidence of useful signal; these are not causal effects.')
st.line_chart(pd.read_csv(ARTIFACTS / 'calibration.csv').set_index('predicted')['observed'], x_label='Predicted probability', y_label='Observed no-show fraction')

st.subheader('Forecast evidence')
specialty = st.selectbox('Forecast to inspect', list(report['demand']), key='evidence_specialty')
f = report['demand'][specialty]
st.dataframe(pd.DataFrame(f['validation_comparison']), hide_index=True)
st.write('Final test metrics', f['test'])
slug = ''.join(char if char.isalnum() else '_' for char in specialty)
st.line_chart(pd.read_csv(ARTIFACTS / f'backtest_{slug}.csv', parse_dates=['date']).set_index('date'))
st.caption('Validation: three expanding 28-day blocks. Test: two subsequent 28-day blocks. Within a block, predicted volumes feed future lags; actual future demand is not supplied.')

st.subheader('Outreach scenario')
capacity = st.selectbox('Share of appointments contacted', [.05, .1, .2, .3], index=2, format_func=lambda x: f'{x:.0%}')
effect = st.slider('Assumed fraction of targeted no-shows prevented', 0.0, .5, .2, .01)
row = next(x for x in c['outreach'] if x['capacity_fraction'] == capacity)
relative = row['share_of_all_no_shows'] * effect
absolute = c['test']['no_show_rate'] * relative
metric_row([('Historical missed visits captured', f"{row['share_of_all_no_shows']:.1%}"),
            ('Scenario relative reduction', f'{relative:.1%}'),
            ('Scenario percentage-point reduction', f'{absolute * 100:.2f}')])
st.caption('This is a scenario based on historical ranking and the effectiveness you assumed. No intervention effect or revenue saving has been measured. A randomised reminder pilot is needed to estimate either.')
overall = report['demand']['All specialties']
baseline_mae = overall['baseline_test']['mae']
if baseline_mae:
    gain = 1 - overall['test']['mae'] / baseline_mae
    st.write(f'Overall forecast MAE improvement versus the weekly baseline: {gain:.1%}. This does not establish a staffing-efficiency gain.')
with st.expander('Methodology and deployment notes'):
    st.markdown((ARTIFACTS / 'REPORT.md').read_text(encoding='utf-8'))
st.download_button('Download full findings report', (ARTIFACTS / 'REPORT.md').read_bytes(), 'CER_results.md', 'text/markdown')

from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_pages_and_prediction_form():
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=90).run()
    assert not at.exception
    at.switch_page('app_pages/risk.py').run()
    assert not at.exception
    next(b for b in at.button if b.label == 'Calculate risk').click().run()
    assert not at.exception
    assert any(m.label == 'Estimated probability of missing the appointment' for m in at.metric)
    at.switch_page('app_pages/demand.py').run()
    assert not at.exception
    at.selectbox(key='forecast_specialty').select('physiotherapy').run()
    assert not at.exception
    assert any(m.label == 'Test MAE' for m in at.metric)
    at.switch_page('app_pages/evidence.py').run()
    assert not at.exception
    assert any(m.label == 'Scenario relative reduction' for m in at.metric)

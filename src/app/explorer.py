import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(page_title="TSB-AD Explorer", layout="wide")

pages = {
    "Benchmark": [
        st.Page("pages/global_summary.py", title="Global Summary"),
        st.Page("pages/edge_cases.py",     title="Edge Cases"),
    ],
    "Series": [
        st.Page("pages/series_browser.py", title="Series Browser"),
    ],
    "Anomalies": [
        st.Page("pages/anomaly_clusters.py", title="Anomaly Clusters"),
    ],
}
st.navigation(pages).run()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import streamlit as st

from app.utils import (
    get_kind, get_parser, cached_metrics, cached_meta,
    VALID_METRICS, DEFAULT_METRIC,
    load_saved, append_saved, plot_series_plotly,
)

st.title("Series Browser")

SAMPLE_SIZE = 20

# ---------------------------------------------------------------------------
# Sidebar — kind first so parser/meta are available for remaining widgets
# ---------------------------------------------------------------------------
kind = get_kind()
parser = get_parser(kind)
meta_df = cached_meta(kind)
datasets = ["All"] + sorted(meta_df['dataset'].unique().tolist())
detectors_available = parser.get_detectors()

with st.sidebar:
    mode = st.radio("Mode", ["Random", "Saved"])
    if mode == "Random":
        dataset_filter = st.selectbox("Dataset", datasets)
    selected_detectors = st.multiselect("Detectors (leave empty to auto-pick top 2)", detectors_available, default=[])
    metric = st.selectbox("Metric", VALID_METRICS, index=VALID_METRICS.index(DEFAULT_METRIC))
    hide_anomalies = st.checkbox("Hide anomalies", value=False)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if st.session_state.get("browser_kind") != kind:
    st.session_state["browser_fnames"] = []
    st.session_state["browser_idx"] = 0
    st.session_state["browser_kind"] = kind
st.session_state.setdefault("browser_fnames", [])
st.session_state.setdefault("browser_idx", 0)

# ---------------------------------------------------------------------------
# Load / sample list of filenames
# ---------------------------------------------------------------------------
if mode == "Random":
    subset = meta_df if dataset_filter == "All" else meta_df[meta_df['dataset'] == dataset_filter]
    if st.button("Sample"):
        all_fnames = list(subset['filename'])
        k = min(SAMPLE_SIZE, len(all_fnames))
        import random
        st.session_state["browser_fnames"] = random.sample(all_fnames, k)
        st.session_state["browser_idx"] = 0
else:
    saved_entries = load_saved()
    st.session_state["browser_fnames"] = [e['filename'] for e in saved_entries]
    st.session_state["browser_idx"] = min(
        st.session_state["browser_idx"],
        max(0, len(st.session_state["browser_fnames"]) - 1),
    )

fnames = st.session_state["browser_fnames"]

if not fnames:
    if mode == "Random":
        st.info("Click **Sample** in the sidebar to load a set of series.")
    else:
        st.info("No saved series yet. Use the Series Browser (Random mode) to save some.")
    st.stop()

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
idx = st.session_state["browser_idx"]
idx = max(0, min(idx, len(fnames) - 1))

col_prev, col_info, col_next = st.columns([1, 6, 1])
with col_prev:
    if st.button("← Prev"):
        st.session_state["browser_idx"] = max(0, idx - 1)
        st.rerun()
with col_info:
    st.write(f"**{idx + 1} / {len(fnames)}**: `{fnames[idx]}`")
with col_next:
    if st.button("Next →"):
        st.session_state["browser_idx"] = min(len(fnames) - 1, idx + 1)
        st.rerun()

# ---------------------------------------------------------------------------
# Load current series
# ---------------------------------------------------------------------------
fname = fnames[idx]

try:
    fnames_loaded, timeseries, labels = parser.load_timeseries(filename=[fname])
except ValueError:
    st.error(f"Could not load: {fname}")
    st.stop()

if not fnames_loaded:
    st.error(f"Could not load: {fname}")
    st.stop()

ts, lab = timeseries[0], labels[0]

# ---------------------------------------------------------------------------
# Determine which detectors to show
# ---------------------------------------------------------------------------
metrics_df = cached_metrics(kind)
npy_fname = fname.replace('.csv', '.npy')

if selected_detectors:
    show_detectors = list(selected_detectors)
else:
    top_rows = (
        metrics_df[
            (metrics_df['File Name'] == fname) &
            (metrics_df['Model Name'].isin(parser.get_detectors()))
        ]
        .sort_values(metric, ascending=False)
        .drop_duplicates(subset='Model Name')
    )
    show_detectors = top_rows['Model Name'].tolist()[:2]

det_scores_list = []
det_names_list = []

for det in show_detectors:
    scores, problematic = parser.load_scores(detectors=[det], filenames=[fname], drop_missing=False)
    if len(scores) == 0 or npy_fname in problematic:
        continue
    det_scores_list.append(scores[0][0])
    row = metrics_df[(metrics_df['File Name'] == fname) & (metrics_df['Model Name'] == det)]
    label = f"{det} ({row[metric].iloc[0]:.3f})" if not row.empty else det
    det_names_list.append(label)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
row_meta = meta_df[meta_df['filename'] == fname]
dataset_label = row_meta['dataset'].iloc[0] if not row_meta.empty else 'Unknown'
entity_label = row_meta['entity_type'].iloc[0] if not row_meta.empty else 'Unknown'

title = f"{fname}  |  Dataset={dataset_label}  |  Type={entity_label}"
fig = plot_series_plotly(
    ts, lab if not hide_anomalies else None,
    title=title,
    detector_scores=det_scores_list or None,
    detector_names=det_names_list or None,
    show_anomalies=not hide_anomalies,
)
st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------------
# Notes + Save
# ---------------------------------------------------------------------------
saved_notes = ""
if mode == "Saved":
    saved_entries = load_saved()
    notes_map = {e['filename']: e['notes'] for e in saved_entries}
    saved_notes = notes_map.get(fname, "")

notes = st.text_area("Notes", value=saved_notes, height=80)

if st.button("Save"):
    append_saved(fname, notes)
    st.success(f"Saved: {fname}")

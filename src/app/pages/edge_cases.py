import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from app.utils import (
    get_kind, get_parser, cached_metrics, cached_meta,
    VALID_METRICS, DEFAULT_METRIC, plot_series_plotly,
)

st.title("Edge Cases")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    kind = get_kind()
    metric = st.selectbox("Metric", VALID_METRICS, index=VALID_METRICS.index(DEFAULT_METRIC))
    perspective = st.radio("Perspective", ["Time Series", "Detectors", "Both"])
    meta_df = cached_meta(kind)
    datasets = ["All"] + sorted(meta_df['dataset'].unique().tolist())
    entity_types = ["All"] + sorted(meta_df['entity_type'].unique().tolist())
    dataset_filter = st.selectbox("Dataset", datasets)
    entity_filter = st.selectbox("Entity Type", entity_types)
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=3, step=1)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
parser = get_parser(kind)
metrics = cached_metrics(kind)
meta = meta_df[['filename', 'dataset', 'entity_type']]

metrics_enriched = metrics.merge(meta, left_on='File Name', right_on='filename', how='left')

if dataset_filter != "All":
    metrics_enriched = metrics_enriched[metrics_enriched['dataset'] == dataset_filter]
if entity_filter != "All":
    metrics_enriched = metrics_enriched[metrics_enriched['entity_type'] == entity_filter]

available_detectors = set(parser.get_detectors())


# ---------------------------------------------------------------------------
# Shared series viewer
# ---------------------------------------------------------------------------
def _show_series(fname, label_prefix, idx_key):
    try:
        fnames_loaded, timeseries, labels = parser.load_timeseries(filename=[fname])
    except ValueError:
        st.warning(f"Could not load {fname}")
        return
    if not fnames_loaded:
        st.warning(f"Could not load {fname}")
        return

    ts, lab = timeseries[0], labels[0]

    best_det_rows = (
        metrics_enriched[
            (metrics_enriched['File Name'] == fname) &
            (metrics_enriched['Model Name'].isin(available_detectors))
        ]
        .sort_values(metric, ascending=False)
        .drop_duplicates(subset='Model Name')
        .head(2)
    )
    det_names_to_load = list(best_det_rows['Model Name'])
    det_vus_map = best_det_rows.set_index('Model Name')[metric].to_dict()

    scores_list = []
    names_list = []
    for det in det_names_to_load:
        scores, problematic = parser.load_scores(detectors=[det], filenames=[fname], drop_missing=False)
        npy = fname.replace('.csv', '.npy')
        if len(scores) == 0 or npy in problematic:
            continue
        scores_list.append(scores[0][0])
        val = det_vus_map.get(det, float('nan'))
        names_list.append(f"{det} ({val:.3f})")

    row = metrics_enriched[metrics_enriched['File Name'] == fname].iloc[0] if not metrics_enriched[metrics_enriched['File Name'] == fname].empty else None
    avg_val = (
        metrics_enriched[metrics_enriched['File Name'] == fname][metric].mean()
    )
    dataset_label = row['dataset'] if row is not None else 'Unknown'
    entity_label = row['entity_type'] if row is not None else 'Unknown'

    title = (
        f"{label_prefix}{fname}  |  Dataset={dataset_label}  |  "
        f"Type={entity_label}  |  Mean {metric}={avg_val:.3f}"
    )
    fig = plot_series_plotly(
        ts, lab, title=title,
        detector_scores=scores_list or None,
        detector_names=names_list or None,
    )
    st.plotly_chart(fig, width='stretch')


def _viewer(fnames_hard, fnames_easy, key_prefix):
    idx_key = f"{key_prefix}_idx"
    combined = [('Hard', f) for f in fnames_hard] + [('Easy', f) for f in fnames_easy]

    if not combined:
        st.info("No series found.")
        return

    st.session_state.setdefault(idx_key, 0)
    idx = st.session_state[idx_key]
    idx = max(0, min(idx, len(combined) - 1))

    col_prev, col_info, col_next = st.columns([1, 6, 1])
    with col_prev:
        if st.button("← Prev", key=f"{key_prefix}_prev"):
            st.session_state[idx_key] = max(0, idx - 1)
            st.rerun()
    with col_info:
        level, fname = combined[idx]
        st.write(f"**{level}** — {idx + 1} / {len(combined)}: `{fname}`")
    with col_next:
        if st.button("Next →", key=f"{key_prefix}_next"):
            st.session_state[idx_key] = min(len(combined) - 1, idx + 1)
            st.rerun()

    level, fname = combined[idx]
    _show_series(fname, f"[{level}] ", key_prefix)


# ---------------------------------------------------------------------------
# Time Series perspective
# ---------------------------------------------------------------------------
def render_timeseries_perspective():
    st.subheader("Time Series Edge Cases")
    series_rank = (
        metrics_enriched
        .groupby('File Name', as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=True)
    )
    if series_rank.empty:
        st.info("No data after filtering.")
        return

    hardest = list(series_rank.head(int(top_k))['File Name'])
    easiest = list(series_rank.tail(int(top_k))['File Name'])
    easiest = [f for f in easiest if f not in hardest]
    _viewer(hardest, easiest, "ts_edge")


# ---------------------------------------------------------------------------
# Detectors perspective
# ---------------------------------------------------------------------------
def render_detectors_perspective():
    st.subheader("Detectors Edge Cases")
    detector_order = (
        metrics_enriched[metrics_enriched['Model Name'].isin(available_detectors)]
        .groupby('Model Name')[metric]
        .mean()
        .sort_values(ascending=False)
    )
    top_detectors = list(detector_order.head(3).index)

    for det in top_detectors:
        st.markdown(f"#### Detector: {det}")
        det_rows = (
            metrics_enriched[metrics_enriched['Model Name'] == det]
            [['File Name', metric]]
            .sort_values(metric, ascending=True)
        )
        hardest = list(det_rows.head(int(top_k))['File Name'])
        easiest = list(det_rows.tail(int(top_k))['File Name'])
        easiest = [f for f in easiest if f not in hardest]
        _viewer(hardest, easiest, f"det_edge_{det}")


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
if perspective in ("Time Series", "Both"):
    render_timeseries_perspective()

if perspective in ("Detectors", "Both"):
    render_detectors_perspective()

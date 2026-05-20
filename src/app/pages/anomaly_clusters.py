import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from app.utils import get_kind, get_parser, cached_meta, plot_series_plotly

st.title("Anomaly Clusters")

# ---------------------------------------------------------------------------
# Kind selector (must come before parquet guard so the error message is accurate)
# ---------------------------------------------------------------------------
kind = get_kind()
parquet_path = Path(f"data/anomaly_features_{kind}.parquet")

# ---------------------------------------------------------------------------
# Parquet guard
# ---------------------------------------------------------------------------
if not parquet_path.exists():
    st.error(f"Run:  python src/app/precompute.py --kind {kind}")
    st.stop()


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_features(kind: str) -> pd.DataFrame:
    return pd.read_parquet(f"data/anomaly_features_{kind}.parquet")


@st.cache_data
def run_pca_kmeans(df_hash: int, n_clusters: int, detector_cols: tuple, kind: str) -> pd.DataFrame:
    df = load_features(kind)
    X = df[list(detector_cols)].fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)
    n_clusters = min(n_clusters, len(df))
    cluster_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_scaled)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return df.assign(PC1=X_pca[:, 0], PC2=X_pca[:, 1], cluster=cluster_labels)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df_all = load_features(kind)
detectors = [c for c in df_all.columns if c not in {
    'filename', 'dataset', 'entity_type', 'anomaly_idx',
    'start', 'end', 'length', 'overall_detectability', 'best_detector',
    'PC1', 'PC2', 'cluster',
}]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    n_clusters = st.slider("Number of clusters", min_value=2, max_value=12, value=6)
    all_datasets = sorted(df_all['dataset'].unique().tolist())
    all_entity_types = sorted(df_all['entity_type'].unique().tolist())
    sel_datasets = st.multiselect("Dataset", all_datasets, default=all_datasets)
    sel_entity_types = st.multiselect("Entity Type", all_entity_types, default=all_entity_types)
    top_k = st.slider("Top K (rank view)", min_value=1, max_value=50, value=10)

# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------
df = df_all[
    df_all['dataset'].isin(sel_datasets) &
    df_all['entity_type'].isin(sel_entity_types)
].reset_index(drop=True)

if df.empty:
    st.info("No data after filtering.")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_rank, tab_group, tab_cluster = st.tabs(["Rank", "Group by Detector", "Cluster"])

# ---------------------------------------------------------------------------
# Rank tab
# ---------------------------------------------------------------------------
with tab_rank:
    df_sorted = df.sort_values('overall_detectability').reset_index(drop=True)
    n = len(df_sorted)
    k = int(top_k)

    # Histogram of detectability
    fig_rank = go.Figure(go.Histogram(
        x=df_sorted['overall_detectability'],
        nbinsx=30,
        marker_color='#aec7e8',
        name='Anomalies',
    ))

    hardest_thresh = df_sorted['overall_detectability'].iloc[min(k - 1, n - 1)]
    easiest_thresh = df_sorted['overall_detectability'].iloc[max(n - k, 0)]

    fig_rank.add_vline(x=hardest_thresh, line_color='red', line_dash='dash',
                       annotation_text=f"Top-{k} hardest", annotation_position="top left")
    fig_rank.add_vline(x=easiest_thresh, line_color='green', line_dash='dash',
                       annotation_text=f"Top-{k} easiest", annotation_position="top right")

    fig_rank.update_layout(
        xaxis_title='Overall Detectability',
        yaxis_title='Count',
        height=350,
        margin=dict(l=60, r=20, t=40, b=40),
        font=dict(size=13),
    )
    st.plotly_chart(fig_rank, width='stretch')

    # Compact tables side by side
    cols_show = ['filename', 'dataset', 'anomaly_idx', 'overall_detectability']
    col_hard, col_easy = st.columns(2)
    with col_hard:
        st.markdown(f"**Top-{k} hardest** (lowest detectability)")
        st.dataframe(
            df_sorted.head(k)[cols_show].reset_index(drop=True),
            use_container_width=True,
        )
    with col_easy:
        st.markdown(f"**Top-{k} easiest** (highest detectability)")
        st.dataframe(
            df_sorted.tail(k).sort_values('overall_detectability', ascending=False)[cols_show].reset_index(drop=True),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Group by Detector tab
# ---------------------------------------------------------------------------
with tab_group:
    col_left, col_right = st.columns(2)

    counts = df['best_detector'].value_counts().reindex(detectors, fill_value=0).sort_values(ascending=False)
    with col_left:
        fig_bar = go.Figure(go.Bar(x=counts.index.tolist(), y=counts.values, marker_color='#636efa'))
        fig_bar.update_layout(
            title='Anomaly count per best-performing detector',
            xaxis_tickangle=70,
            height=400,
            margin=dict(b=120),
            font=dict(size=13),
        )
        st.plotly_chart(fig_bar, width='stretch')

    heatmap_data = (
        df.groupby('best_detector')[detectors]
        .mean()
        .reindex(detectors)
        .dropna(how='all')
    )
    with col_right:
        fig_hm = go.Figure(go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns.tolist(),
            y=heatmap_data.index.tolist(),
            colorscale='YlGnBu',
            text=[[f"{v:.2f}" if pd.notna(v) else "" for v in row] for row in heatmap_data.values],
            texttemplate="%{text}",
        ))
        fig_hm.update_layout(
            title='Mean detectability per group × detector',
            height=400,
            margin=dict(l=120, b=80),
            xaxis_tickangle=70,
            font=dict(size=13),
        )
        st.plotly_chart(fig_hm, width='stretch')

# ---------------------------------------------------------------------------
# Cluster tab
# ---------------------------------------------------------------------------
with tab_cluster:
    df_clustered = run_pca_kmeans(hash(tuple(df.index.tolist())), n_clusters, tuple(detectors), kind)
    df_clustered = df_clustered[
        df_clustered['dataset'].isin(sel_datasets) &
        df_clustered['entity_type'].isin(sel_entity_types)
    ].reset_index(drop=True)

    col_scatter, col_heatmap = st.columns(2)

    with col_scatter:
        unique_clusters = sorted(df_clustered['cluster'].unique())
        scatter_traces = []
        for c in unique_clusters:
            mask = df_clustered['cluster'] == c
            sub = df_clustered[mask]
            scatter_traces.append(go.Scatter(
                x=sub['PC1'], y=sub['PC2'],
                mode='markers',
                marker=dict(size=6, opacity=0.7),
                name=f"Cluster {c}",
                customdata=sub.index.tolist(),
            ))
        fig_scatter = go.Figure(data=scatter_traces)
        fig_scatter.update_layout(
            title='Anomalies in PCA space',
            height=450,
            margin=dict(l=40, r=20, t=50, b=40),
            font=dict(size=13),
        )
        event = st.plotly_chart(
            fig_scatter,
            on_select="rerun",
            selection_mode="points",
            key="cluster_scatter",
            width='stretch',
        )

    with col_heatmap:
        cluster_means = df_clustered.groupby('cluster')[detectors].mean()
        fig_chm = go.Figure(go.Heatmap(
            z=cluster_means.values,
            x=cluster_means.columns.tolist(),
            y=[f"Cluster {c}" for c in cluster_means.index],
            colorscale='YlGnBu',
            text=[[f"{v:.2f}" if pd.notna(v) else "" for v in row] for row in cluster_means.values],
            texttemplate="%{text}",
        ))
        fig_chm.update_layout(
            title='Mean detectability per cluster × detector',
            height=450,
            margin=dict(l=80, b=80),
            xaxis_tickangle=70,
            font=dict(size=13),
        )
        st.plotly_chart(fig_chm, width='stretch')

    # ---------------------------------------------------------------------------
    # Click-to-inspect
    # ---------------------------------------------------------------------------
    selection = event.selection if event else None
    if selection and selection.get("points"):
        point = selection["points"][0]
        row_idx = point.get("point_index", None)
        if row_idx is not None and row_idx < len(df_clustered):
            row = df_clustered.iloc[row_idx]

            st.divider()
            st.subheader("Inspecting selected anomaly")

            parser = get_parser(kind)
            try:
                fnames_loaded, timeseries_loaded, labels_loaded = parser.load_timeseries(filename=[row['filename']])
            except ValueError:
                st.error(f"Could not load {row['filename']}")
                st.stop()

            ts_full, lab_full = timeseries_loaded[0], labels_loaded[0]
            T = len(ts_full)

            ctx_start = int(max(0, row['start'] - 100))
            ctx_end = int(min(T, row['end'] + 100))
            ts_ctx = ts_full[ctx_start:ctx_end] if ts_full.ndim == 1 else ts_full[ctx_start:ctx_end, :]
            lab_ctx = lab_full[ctx_start:ctx_end]

            det_vals = {d: row[d] for d in detectors if pd.notna(row.get(d))}
            top2_dets = sorted(det_vals, key=lambda d: det_vals[d], reverse=True)[:2]

            det_scores_ctx = []
            det_names_ctx = []
            for det in top2_dets:
                scores, problematic = parser.load_scores(
                    detectors=[det], filenames=[row['filename']], drop_missing=False
                )
                npy = row['filename'].replace('.csv', '.npy')
                if len(scores) == 0 or npy in problematic:
                    continue
                full_score = scores[0][0]
                det_scores_ctx.append(full_score[ctx_start:ctx_end])
                det_names_ctx.append(f"{det} ({det_vals[det]:.3f})")

            fig_detail = plot_series_plotly(
                ts_ctx, lab_ctx,
                title=f"{row['filename']}  |  Anomaly #{int(row['anomaly_idx'])}  |  Cluster {int(row['cluster'])}",
                detector_scores=det_scores_ctx or None,
                detector_names=det_names_ctx or None,
                context_start=ctx_start,
            )
            st.plotly_chart(fig_detail, width='stretch')

            meta_info = {
                'filename': row['filename'],
                'dataset': row['dataset'],
                'entity_type': row['entity_type'],
                'anomaly_idx': int(row['anomaly_idx']),
                'start': int(row['start']),
                'end': int(row['end']),
                'length': int(row['length']),
                'cluster': int(row['cluster']),
                'overall_detectability': float(row['overall_detectability']),
            }
            st.table(pd.DataFrame([meta_info]).T.rename(columns={0: 'value'}).astype(str))

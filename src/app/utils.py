import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from tsbadparser import TSBADParser


VALID_METRICS = [
    'AUC-PR',
    'VUS-PR (L: 0)', 'VUS-PR (L: 1)', 'VUS-PR (L: 2)', 'VUS-PR (L: 4)',
    'VUS-PR (L: 8)', 'VUS-PR (L: 16)', 'VUS-PR (L: 32)', 'VUS-PR (L: 64)',
    'VUS-PR (L: 128)', 'VUS-PR (L: 256)',
]
DEFAULT_METRIC = 'VUS-PR (L: 128)'

THRESHOLDS = np.linspace(0, 1, 21)
BUFFERS = [0, 4, 16, 64, 128]

SAVED_FILE = Path(__file__).parent.parent.parent / 'saved_series.json'


# ---------------------------------------------------------------------------
# Kind selector + cached parser + data loaders
# ---------------------------------------------------------------------------

def get_kind() -> str:
    return st.sidebar.selectbox("Kind", ["uni", "multi"], key="kind_selector")


@st.cache_resource
def get_parser(kind: str = 'uni') -> TSBADParser:
    return TSBADParser(path='data', kind=kind)


@st.cache_data
def cached_metrics(kind: str = 'uni') -> pd.DataFrame:
    return get_parser(kind).load_metrics(models=get_parser(kind).get_all_models())


@st.cache_data
def cached_runtime(kind: str = 'uni') -> pd.DataFrame:
    return get_parser(kind).load_runtime(models=get_parser(kind).get_all_models())


@st.cache_data
def cached_meta(kind: str = 'uni') -> pd.DataFrame:
    return get_parser(kind).meta.reset_index()


# ---------------------------------------------------------------------------
# Anomaly helpers (ported from review_anomalies.py)
# ---------------------------------------------------------------------------

def extract_anomaly_segments(labels):
    """Return list of (start, end) for each contiguous anomaly segment."""
    diff = np.diff(np.concatenate([[0], labels.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def compute_detectability(labels, score, thresholds=THRESHOLDS, buffers=BUFFERS):
    """Per-anomaly TP density for one detector score array.

    Returns:
        detectabilities: ndarray of shape (n_anomalies,)
        segments:        list of (start, end) tuples
    """
    segments = extract_anomaly_segments(labels)
    T = len(labels)
    detectabilities = []
    for start, end in segments:
        tp_fracs = []
        for buf in buffers:
            pad_start = max(0, start - buf)
            pad_end = min(T, end + buf)
            pad_len = pad_end - pad_start
            for thr in thresholds:
                pred = (score >= thr).astype(int)
                tp = pred[pad_start:pad_end].sum()
                tp_fracs.append(tp / pad_len if pad_len > 0 else 0.0)
        detectabilities.append(float(np.mean(tp_fracs)))
    return np.array(detectabilities), segments


def build_anomaly_table(parser, fnames, timeseries, labels, all_scores, detectors):
    """Build a DataFrame with one row per anomaly and per-detector detectability features."""
    from tqdm import tqdm
    rows = []
    for fname, ts, lab, file_scores in tqdm(
        zip(fnames, timeseries, labels, all_scores), total=len(fnames), desc="Processing time series"
    ):
        segments = extract_anomaly_segments(lab)
        if not segments:
            continue

        det_scores = {}
        for d_idx, det in enumerate(detectors):
            d_det, _ = compute_detectability(lab, file_scores[d_idx])
            det_scores[det] = d_det

        meta_row = parser.meta[parser.meta.filename == fname]
        dataset = meta_row['dataset'].iloc[0] if not meta_row.empty else 'Unknown'
        entity_type = meta_row['entity_type'].iloc[0] if not meta_row.empty else 'Unknown'

        for j, (start, end) in enumerate(segments):
            row = {
                'filename': fname,
                'dataset': dataset,
                'entity_type': entity_type,
                'anomaly_idx': j,
                'start': start,
                'end': end,
                'length': end - start,
            }
            for det in detectors:
                row[det] = det_scores[det][j] if j < len(det_scores[det]) else 0.0
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df['overall_detectability'] = df[detectors].mean(axis=1)
        df['best_detector'] = df[detectors].idxmax(axis=1)
    return df


# ---------------------------------------------------------------------------
# Saved-series helpers (ported from review_random.py)
# ---------------------------------------------------------------------------

def load_saved() -> list[dict]:
    """Load saved entries. Migrates old list-of-strings format transparently."""
    if SAVED_FILE.exists():
        with open(SAVED_FILE, 'r') as f:
            data = json.load(f)
        if data and isinstance(data[0], str):
            data = [{'filename': s, 'notes': ''} for s in data]
        return data
    return []


def append_saved(fname: str, notes: str = '') -> None:
    """Upsert a saved entry. Updates notes if the series was already saved."""
    saved = load_saved()
    for entry in saved:
        if entry['filename'] == fname:
            entry['notes'] = notes
            with open(SAVED_FILE, 'w') as f:
                json.dump(saved, f, indent=2)
            return
    saved.append({'filename': fname, 'notes': notes})
    with open(SAVED_FILE, 'w') as f:
        json.dump(saved, f, indent=2)


# ---------------------------------------------------------------------------
# Plotly time-series helper
# ---------------------------------------------------------------------------

def plot_series_plotly(
    timeseries: np.ndarray,
    anomaly_labels: np.ndarray | None,
    title: str = "",
    detector_scores: list[np.ndarray] | None = None,
    detector_names: list[str] | None = None,
    context_start: int = 0,
    show_anomalies: bool = True,
) -> go.Figure:
    ts_2d = timeseries if timeseries.ndim == 2 else timeseries.reshape(-1, 1)
    n_channels = ts_2d.shape[1]
    n_detectors = len(detector_scores) if detector_scores else 0
    n_rows = n_channels + n_detectors

    row_heights = [3] * n_channels + [1] * n_detectors
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.02,
    )

    T = ts_2d.shape[0]
    x = np.arange(context_start, context_start + T)

    for ch in range(n_channels):
        fig.add_trace(
            go.Scatter(x=x, y=ts_2d[:, ch], mode='lines', line=dict(width=1.2, color='#1f77b4'),
                       name=f'C{ch}' if n_channels > 1 else 'Series', showlegend=False),
            row=ch + 1, col=1,
        )

        if anomaly_labels is not None and show_anomalies:
            segments = extract_anomaly_segments(anomaly_labels)
            for seg_start, seg_end in segments:
                seg_x = x[seg_start:seg_end]
                seg_y = ts_2d[seg_start:seg_end, ch]
                if len(seg_x) == 1:
                    fig.add_trace(
                        go.Scatter(x=seg_x, y=seg_y, mode='markers',
                                   marker=dict(size=6, color='red'), showlegend=False),
                        row=ch + 1, col=1,
                    )
                else:
                    fig.add_trace(
                        go.Scatter(x=seg_x, y=seg_y, mode='lines', line=dict(width=1.5, color='red'),
                                   showlegend=False),
                        row=ch + 1, col=1,
                    )

    for di, score in enumerate(detector_scores or []):
        name = (detector_names[di] if detector_names else f'D{di}')
        row_idx = n_channels + di + 1
        color = '#e07b00' if di == 0 else None
        fig.add_trace(
            go.Scatter(x=x, y=score, mode='lines', line=dict(width=1.2, color=color) if color else dict(width=1.2),
                       name=name, showlegend=True),
            row=row_idx, col=1,
        )
        fig.update_yaxes(range=[-0.02, 1.02], title_text='', row=row_idx, col=1)

        axis_key = 'yaxis' if row_idx == 1 else f'yaxis{row_idx}'
        domain = fig.layout[axis_key].domain
        y_center = (domain[0] + domain[1]) / 2

        if ' (' in name:
            det_label, val_part = name.rsplit(' (', 1)
            ann_text = f"{det_label}<br>({val_part}"
        else:
            ann_text = name
        fig.add_annotation(
            text=ann_text,
            xref='paper', yref='paper',
            x=-0.02, y=y_center,
            xanchor='right', yanchor='middle',
            textangle=-90,
            showarrow=False,
            font=dict(size=11),
        )

    for ch in range(n_channels):
        label = f'C{ch}' if n_channels > 1 else 'Value'
        fig.update_yaxes(title_text=label, row=ch + 1, col=1)

    fig.update_layout(
        title=title,
        height=max(300, 150 * n_rows),
        margin=dict(l=80, r=20, t=50, b=40),
        font=dict(size=13),
    )
    fig.update_xaxes(title_text='Time', row=n_rows, col=1)
    return fig

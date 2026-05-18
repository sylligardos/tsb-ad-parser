"""
Smoke tests and usage examples for TSBADParser.

Run all:           pytest -v -s tests/test_parser.py
Run one section:   pytest -v -s tests/test_parser.py -k "timeseries"
"""

import numpy as np
import pytest
from tsbadparser import TSBADParser


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_meta_shape(parser_uni, parser_multi):
    for parser in (parser_uni, parser_multi):
        assert len(parser.meta) > 0, "Metadata table is empty"
        expected_cols = {"dataset", "id_in_dataset", "entity_type", "train_length", "first_anomaly", "filename"}
        assert expected_cols.issubset(parser.meta.columns)
        print(f"\n[{parser.kind}] {len(parser.meta)} time series in metadata")


def test_get_filenames(parser_uni, parser_multi):
    for parser in (parser_uni, parser_multi):
        fnames = parser.get_filenames()
        assert len(fnames) > 0
        assert all(f.endswith(".csv") for f in fnames)
        print(f"[{parser.kind}] {len(fnames)} files")


def test_get_datasets(parser_uni, parser_multi):
    for parser in (parser_uni, parser_multi):
        datasets = parser.get_datasets()
        assert len(datasets) > 0
        assert datasets == sorted(datasets), "Datasets not sorted"
        print(f"[{parser.kind}] datasets: {datasets}")


# ---------------------------------------------------------------------------
# Model lists
# ---------------------------------------------------------------------------

def test_get_detectors(parser_uni, parser_multi):
    for parser in (parser_uni, parser_multi):
        detectors = parser.get_detectors()
        assert len(detectors) > 0
        assert detectors == sorted(detectors)
        print(f"[{parser.kind}] {len(detectors)} detectors: {detectors[:3]} ...")


def test_get_baselines(parser_uni):
    baselines = parser_uni.get_baselines()
    assert "Oracle" in baselines
    assert "RandomMS" in baselines
    print(f"Baselines: {baselines}")


def test_get_ensembles(parser_uni, parser_multi):
    for parser in (parser_uni, parser_multi):
        ensembles = parser.get_ensembles()
        assert len(ensembles) > 0
        # Ensembles should not overlap with detectors
        overlap = set(ensembles) & set(parser.get_detectors())
        assert len(overlap) == 0, f"Ensemble/detector overlap: {overlap}"
        print(f"[{parser.kind}] ensembles: {ensembles}")


def test_get_all_models(parser_uni, parser_multi):
    for parser in (parser_uni, parser_multi):
        all_models = parser.get_all_models()
        detectors = set(parser.get_detectors())
        ensembles = set(parser.get_ensembles())
        baselines = set(parser.get_baselines())
        assert detectors.issubset(all_models)
        assert ensembles.issubset(all_models)
        assert baselines.issubset(all_models)
        print(f"[{parser.kind}] {len(all_models)} total models")


def test_detector_ensemble_baseline_are_disjoint(parser_uni, parser_multi):
    """The three categories must never overlap; if they do, load_scores would
    load ensembles when the caller expects only detectors, or vice versa."""
    for parser in (parser_uni, parser_multi):
        detectors = set(parser.get_detectors())
        ensembles = set(parser.get_ensembles())
        baselines = set(parser.get_baselines())
        assert detectors & ensembles == set(), f"[{parser.kind}] detector/ensemble overlap: {detectors & ensembles}"
        assert detectors & baselines == set(), f"[{parser.kind}] detector/baseline overlap: {detectors & baselines}"
        assert ensembles & baselines == set(), f"[{parser.kind}] ensemble/baseline overlap: {ensembles & baselines}"
        print(f"[{parser.kind}] detectors={len(detectors)}, ensembles={len(ensembles)}, baselines={len(baselines)}, all disjoint ✓")


def test_all_score_dirs_are_classified(parser_uni, parser_multi):
    """Every subdirectory in scores/<kind>/ must belong to exactly one category.
    This test fails if you add new score files without classifying them."""
    for parser in (parser_uni, parser_multi):
        all_models = set(parser.get_all_models())
        score_dirs = {
            name for name in parser.scores_path.iterdir() if name.is_dir()
        }
        score_dir_names = {d.name for d in score_dirs}
        unclassified = score_dir_names - all_models
        assert unclassified == set(), (
            f"[{parser.kind}] Unclassified score directories: {sorted(unclassified)}\n"
            "Add them to DETECTORS, ENSEMBLES, or BASELINES."
        )
        print(f"[{parser.kind}] All {len(score_dir_names)} score dirs are classified ✓")


def test_all_detectors_have_score_dirs(parser_uni, parser_multi):
    """Every listed detector must have a corresponding scores/<kind>/<name>/
    directory, otherwise load_scores would silently skip it."""
    for parser in (parser_uni, parser_multi):
        score_dir_names = {d.name for d in parser.scores_path.iterdir() if d.is_dir()}
        detectors = set(parser.get_detectors())
        missing = detectors - score_dir_names
        assert missing == set(), (
            f"[{parser.kind}] Detectors with no score directory: {sorted(missing)}"
        )
        print(f"[{parser.kind}] All {len(detectors)} detectors have score dirs ✓")


def test_load_scores_default_loads_detectors_only(parser_uni):
    """load_scores() with no arguments must not include ensemble scores."""
    fnames = parser_uni.get_filenames()[:5]
    scores, _ = parser_uni.load_scores(filenames=fnames)
    n_detectors = len(parser_uni.get_detectors())
    for score_matrix in scores:
        assert score_matrix.shape[0] == n_detectors, (
            f"Expected {n_detectors} detector rows, got {score_matrix.shape[0]}"
        )
    print(f"\nDefault load_scores: {n_detectors} detectors per time series ✓")


# ---------------------------------------------------------------------------
# Default path
# ---------------------------------------------------------------------------

def test_default_path_works():
    """Parser constructed with no path argument should find data/ automatically."""
    parser = TSBADParser(kind="uni")
    assert len(parser.meta) > 0
    print(f"\nDefault path resolved to: {parser.path}")


def test_bad_path_raises():
    with pytest.raises(FileNotFoundError):
        TSBADParser(path="/nonexistent/path/to/data")


# ---------------------------------------------------------------------------
# Load time series
# ---------------------------------------------------------------------------

def test_load_timeseries_all(parser_uni):
    fnames, timeseries, labels = parser_uni.load_timeseries(njobs=4)
    assert len(fnames) == len(timeseries) == len(labels)
    assert len(fnames) > 0
    for ts, lbl in zip(timeseries[:5], labels[:5]):
        assert ts.ndim in (1, 2)
        assert lbl.ndim == 1
        assert ts.shape[0] == lbl.shape[0], "Time series and label length mismatch"
    print(f"\nLoaded {len(fnames)} uni time series")


def test_load_timeseries_by_dataset(parser_uni):
    dataset = parser_uni.get_datasets()[0]
    fnames, timeseries, labels = parser_uni.load_timeseries(dataset=[dataset])
    assert len(fnames) > 0
    # All returned files must belong to the requested dataset
    parsed = parser_uni.parse_ts_filenames(fnames)
    assert (parsed.dataset == dataset).all(), "Got files from wrong dataset"
    print(f"\nDataset '{dataset}': {len(fnames)} time series")


def test_load_timeseries_by_series_id(parser_uni):
    target_ids = [1, 2, 3]
    fnames, timeseries, labels = parser_uni.load_timeseries(series_id=target_ids)
    assert len(fnames) == len(target_ids)
    parsed = parser_uni.parse_ts_filenames(fnames)
    assert set(parsed.index.tolist()) == set(target_ids)
    print(f"\nLoaded series {target_ids}: {fnames}")


def test_load_timeseries_bad_selector(parser_uni):
    with pytest.raises(ValueError, match="exactly ONE"):
        parser_uni.load_timeseries(dataset=["NAB"], series_id=[1])


def test_load_timeseries_no_match(parser_uni):
    with pytest.raises(ValueError, match="No matching"):
        parser_uni.load_timeseries(dataset=["NONEXISTENT_DATASET_XYZ"], load_all=False)


# ---------------------------------------------------------------------------
# Load scores
# ---------------------------------------------------------------------------

def test_load_scores_two_detectors(parser_uni):
    detectors = ["IForest", "LOF"]
    fnames = parser_uni.get_filenames()[:10]
    scores, bad = parser_uni.load_scores(detectors=detectors, filenames=fnames)
    assert len(scores) > 0
    for score_matrix in scores:
        assert score_matrix.ndim == 2
        assert score_matrix.shape[0] == len(detectors), "Wrong number of detector rows"
    print(f"\nScores loaded for {len(scores)} series; {len(bad)} problematic")


def test_load_scores_shape_consistency(parser_uni):
    detectors = ["IForest", "LOF"]
    fnames, timeseries, labels = parser_uni.load_timeseries(series_id=[1, 2, 3])
    scores, bad = parser_uni.load_scores(detectors=detectors, filenames=fnames)
    kept_fnames = [f for f in fnames if f.replace(".csv", ".npy") not in bad]
    for fname, ts, lbl, score in zip(kept_fnames, timeseries, labels, scores):
        assert score.shape[1] == ts.shape[0], f"Score length mismatch for {fname}"


# ---------------------------------------------------------------------------
# load_scores_and_delete
# ---------------------------------------------------------------------------

def test_load_scores_and_delete(parser_uni):
    detectors = ["IForest", "LOF"]
    fnames, timeseries, labels = parser_uni.load_timeseries(series_id=list(range(1, 11)))
    n_before = len(fnames)
    scores, bad = parser_uni.load_scores_and_delete(
        fnames, timeseries, labels, detectors=detectors
    )
    # All lists must be aligned after the call
    assert len(fnames) == len(timeseries) == len(labels) == len(scores)
    assert len(fnames) <= n_before
    print(f"\nAfter delete: {len(fnames)} series kept, {len(bad)} dropped")


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_load_splits(parser_uni, parser_multi):
    for parser in (parser_uni, parser_multi):
        splits = parser.load_splits()
        assert "eval" in splits and "label" in splits
        assert len(splits["eval"]) > 0
        assert len(splits["label"]) > 0
        # No overlap between eval and label sets
        overlap = set(splits["eval"]) & set(splits["label"])
        assert len(overlap) == 0, f"Eval/label split overlap: {overlap}"
        print(f"[{parser.kind}] eval={len(splits['eval'])}, label={len(splits['label'])}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_load_metrics(parser_uni):
    detectors = parser_uni.get_detectors()[:3]
    df = parser_uni.load_metrics(models=detectors)
    assert not df.empty
    assert "file_name" in df.columns or len(df) > 0
    print(f"\nMetrics shape: {df.shape}")
    print(df.head(3).to_string())


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

def test_load_runtime(parser_uni):
    detectors = parser_uni.get_detectors()[:3]
    df = parser_uni.load_runtime(models=detectors)
    assert not df.empty
    print(f"\nRuntime shape: {df.shape}")
    print(df.head(3).to_string())


# ---------------------------------------------------------------------------
# Visualization (opens a window, close it to continue)
# ---------------------------------------------------------------------------

def test_visualize_timeseries(parser_uni):
    fnames, timeseries, labels = parser_uni.load_timeseries(series_id=[1])
    scores, bad = parser_uni.load_scores(
        detectors=["IForest", "LOF"],
        filenames=fnames,
    )
    if len(scores) == 0:
        pytest.skip("No scores available for series 1")

    print(f"\nVisualizing {fnames[0]}, close the window to continue.")
    parser_uni.visualize_timeseries(
        timeseries[0],
        anomaly_labels=labels[0],
        title=fnames[0],
        detector_scores=[scores[0][0], scores[0][1]],
        detector_names=["IForest", "LOF"],
    )


def test_visualize_no_labels(parser_uni):
    fnames, timeseries, labels = parser_uni.load_timeseries(series_id=[1])
    print(f"\nVisualizing {fnames[0]} without labels, close the window to continue.")
    parser_uni.visualize_timeseries(timeseries[0], title="No labels")


def test_visualize_multivariate(parser_multi):
    fnames, timeseries, labels = parser_multi.load_timeseries(series_id=[1])
    print(f"\nVisualizing multivariate {fnames[0]}, close the window to continue.")
    parser_multi.visualize_timeseries(
        timeseries[0],
        anomaly_labels=labels[0],
        title=fnames[0],
    )

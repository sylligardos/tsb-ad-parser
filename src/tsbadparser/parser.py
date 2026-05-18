"""
@who: Emmanouil Sylligardos (Sylli)
@where: Paris
@when: 2026 (3rd year PhD)
@what: TSB-AD parser
"""

import os
from pathlib import Path
from multiprocessing import Pool
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class TSBADParser:
    """Parser for the TSB-AD benchmark dataset.

    Provides access to raw time series, anomaly scores, evaluation metrics,
    runtimes, and train/eval splits for both univariate ('uni') and
    multivariate ('multi') variants of the benchmark.

    Args:
        path: Root directory of the benchmark data (the folder that contains
              raw/, scores/, metrics/, runtime/, and splits/). Defaults to
              the data/ directory next to this repo (TSB-AD/data/).
        kind: Either 'uni' (univariate) or 'multi' (multivariate).
    """

    VALID_KINDS = {"uni", "multi"}
    PATTERN = re.compile(
        r"(?P<series_id>\d+)_(?P<dataset>\w+)_id_(?P<id_in_dataset>\d+)_(?P<entity>\w+)_tr_(?P<train>\d+)_1st_(?P<first>\d+)"
    )
    BASELINES = ["Oracle", "RandomMS"]
    # Known ensemble methods. Any scores/<kind>/ subdirectory not in DETECTORS
    # or BASELINES is also auto-discovered as an ensemble at runtime, so this
    # list acts as documentation and a fallback — you don't need to update it
    # when you add a new ensemble directory.
    ENSEMBLES = ["AccuCopy", "AOM", "AverageEnsembling", "CRH", "MaximumEnsembling", "MOA"]
    DETECTORS = [
        "AnomalyTransformer", "AutoEncoder", "CNN", "Chronos", "Donut", "FITS",
        "IForest", "KMeansAD_U", "KShapeAD", "LOF", "LSTMAD", "Lag_Llama",
        "MOMENT_FT", "MOMENT_ZS", "MatrixProfile", "OFA", "OmniAnomaly", "POLY",
        "SAND", "SR", "Series2Graph", "Sub_HBOS", "Sub_IForest", "Sub_KNN",
        "Sub_LOF", "Sub_MCD", "Sub_OCSVM", "Sub_PCA", "TimesFM", "TimesNet",
        "TranAD", "USAD",
    ]
    DETECTORS_MULTI = [
        "AnomalyTransformer", "AutoEncoder", "CBLOF", "CNN", "COPOD", "Donut",
        "EIF", "FITS", "HBOS", "IForest", "KMeansAD", "KNN", "LOF", "LSTMAD",
        "MCD", "OCSVM", "OFA", "OmniAnomaly", "PCA", "RobustPCA", "TimesNet",
        "TranAD", "USAD",
    ]

    # Default data root: repo_root/data/ (this file lives at src/tsbadparser/parser.py)
    _DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent / "data"

    def __init__(self, path: str | Path | None = None, kind: str = "uni"):
        if path is None:
            path = self._DEFAULT_DATA_PATH
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.path}\n"
                "Pass the correct path explicitly: TSBADParser('/your/path/to/data')"
            )

        if kind not in self.VALID_KINDS:
            raise ValueError(f"Unknown kind {kind!r}. Expected one of {self.VALID_KINDS}")
        self.kind = kind

        self.raw_path = self.path / "raw" / self.kind
        self.runtime_path = self.path / "runtime" / self.kind
        self.scores_path = self.path / "scores" / self.kind
        self.splits_path = self.path / "splits"
        self.metrics_path = self.path / "metrics" / self.kind

        self.meta = self.parse_ts_filenames()

    def get_filenames(self) -> list[str]:
        """Return all raw CSV filenames for the current kind.

        Returns:
            List of filenames (not full paths) found in raw/<kind>/.
        """
        return [f for f in os.listdir(self.raw_path) if f.endswith(".csv")]

    def get_datasets(self, sort: bool = True) -> list[str]:
        """Return the unique dataset names present in the benchmark.

        Args:
            sort: Whether to return the names in alphabetical order.

        Returns:
            List of dataset name strings (e.g. ['NAB', 'YAHOO', ...]).
        """
        datasets = list(self.meta.dataset.unique())
        return sorted(datasets) if sort else datasets

    def parse_ts_filenames(self, filenames: Optional[list[str]] = None) -> pd.DataFrame:
        """Parse the structured TSB-AD filename convention into a metadata table.

        Each filename encodes series_id, dataset, id_in_dataset, entity type,
        train length, and first anomaly position. Calling with no arguments
        parses all files under raw/<kind>/.

        Args:
            filenames: Optional list of filenames to parse. Defaults to all
                       files returned by get_filenames().

        Returns:
            DataFrame indexed by series_id with columns: dataset,
            id_in_dataset, entity_type, train_length, first_anomaly, filename.
        """
        rows = []
        filenames = filenames if filenames is not None else self.get_filenames()

        for name in filenames:
            stem = Path(name).stem
            match = self.PATTERN.match(stem)

            if not match:
                raise ValueError(f"Unexpected filename format: {name}")

            rows.append(
                {
                    "series_id": int(match["series_id"]),
                    "dataset": match["dataset"],
                    "id_in_dataset": int(match["id_in_dataset"]),
                    "entity_type": match["entity"],
                    "train_length": int(match["train"]),
                    "first_anomaly": int(match["first"]),
                    "filename": name,
                }
            )

        return pd.DataFrame(rows).sort_values("series_id").set_index("series_id")

    def load_timeseries(
        self,
        filename: list[str] = [],
        dataset: list[str] = [],
        series_id: list[int] = [],
        entity_type: list[str] = [],
        load_all: bool = True,
        njobs: Optional[int] = None,
    ) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
        """Load raw time series and their anomaly labels.

        Pass exactly one selector to filter, or use load_all=True (default)
        to load the full benchmark.

        Args:
            filename:    Filter by exact filename(s) (e.g. ['001_NAB_...']).
            dataset:     Filter by dataset name(s) (e.g. ['NAB', 'YAHOO']).
            series_id:   Filter by integer series ID(s).
            entity_type: Filter by entity type string(s) (e.g. ['Sensor']).
            load_all:    If True and no selector is given, loads everything.
            njobs:       Number of parallel worker processes. None = cpu_count.

        Returns:
            Tuple of (filenames, timeseries, labels) where each element is a
            list of equal length. timeseries[i] and labels[i] are numpy arrays
            corresponding to filenames[i].
        """
        selectors = [filename, dataset, series_id, entity_type]
        args_sum = sum(len(x) > 0 for x in selectors)
        if load_all and not args_sum:
            rows = self.meta
        else:
            if args_sum > 1:
                raise ValueError("Provide exactly ONE selector or use load_all=True.")

            if len(filename):
                rows = self.meta[self.meta.filename.isin(filename)]
            elif len(dataset):
                rows = self.meta[self.meta.dataset.isin(dataset)]
            elif len(series_id):
                rows = self.meta.loc[series_id]
            elif len(entity_type):
                rows = self.meta[self.meta.entity_type.isin(entity_type)]

        if len(rows) == 0:
            raise ValueError("No matching time series found.")

        paths = [str(self.raw_path / f) for f in rows.filename]

        with Pool(processes=njobs) as pool:
            results = list(tqdm(pool.imap(self._load_single_csv, paths), total=len(paths), desc="Loading time series"))
        timeseries, labels = zip(*results)
        return list(rows.filename), list(timeseries), list(labels)

    def _load_single_csv(self, path: Path | str, as_numpy: bool = True):
        df = pd.read_csv(path)

        values = df.iloc[:, :-1]
        labels = df.iloc[:, -1]

        if as_numpy:
            values = values.to_numpy()
            labels = labels.to_numpy().squeeze()
            if values.ndim == 2 and values.shape[1] == 1:
                values = values.squeeze()

        return values, labels

    def _load_single_npy(self, path: Path | str) -> Optional[np.ndarray]:
        try:
            return np.load(path)
        except Exception:
            return None

    def visualize_timeseries(
        self,
        timeseries: np.ndarray,
        anomaly_labels: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        detector_scores: Optional[Sequence[np.ndarray]] = None,
        detector_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Plot a time series with optional anomaly highlights and detector scores.

        Creates a stacked figure: one subplot per channel of the time series,
        followed by one subplot per detector score. Anomalous regions are
        overplotted in red.

        Args:
            timeseries:      1-D or 2-D numpy array of shape (T,) or (T, C).
            anomaly_labels:  Binary array of shape (T,). 1 = anomaly. Optional.
            title:           Figure title. Optional.
            detector_scores: Sequence of 1-D score arrays, one per detector.
                             Must be the same length as timeseries. Optional.
            detector_names:  Names for each detector score subplot. Must match
                             the length of detector_scores if provided.
        """
        if detector_scores is not None and detector_names is not None:
            if len(detector_scores) != len(detector_names):
                raise ValueError(
                    f"Number of detectors ({len(detector_scores)}) does not match "
                    f"number of detector names ({len(detector_names)})"
                )

        ts_2d = timeseries if timeseries.ndim == 2 else timeseries.reshape(-1, 1)
        n_ts_channels = ts_2d.shape[1]
        n_detectors = len(detector_scores) if detector_scores is not None else 0
        n_total_plots = n_ts_channels + n_detectors

        sns.set_style("whitegrid", {"grid.color": "#dddddd"})
        fig, axes = plt.subplots(n_total_plots, 1, figsize=(15, min(4 * n_total_plots, 12)), sharex=True)
        axes = np.atleast_1d(axes)

        for i in range(n_ts_channels):
            sns.lineplot(ts_2d[:, i], ax=axes[i], linewidth=1.5)
            axes[i].set_ylabel(f"C{i}" if n_ts_channels > 1 else "Time series")
        axes[-1].set_xlabel("Time")

        if anomaly_labels is not None:
            if len(anomaly_labels) != len(ts_2d):
                raise ValueError("Label length must match time series length.")

            label_diff = np.diff(anomaly_labels)
            starts = np.where(label_diff == 1)[0]
            ends = np.where(label_diff == -1)[0]

            if len(starts) > 0 and len(ends) > 0:
                if ends[0] < starts[0]:
                    starts = np.r_[0, starts]
                if starts[-1] > ends[-1]:
                    ends = np.r_[ends, len(ts_2d)]

                for start, end in zip(starts, ends):
                    if end - start <= 1:
                        end += 1
                        start -= 1
                    for i in range(n_ts_channels):
                        sns.lineplot(
                            x=np.arange(start, end),
                            y=ts_2d[start:end, i],
                            ax=axes[i],
                            color="red",
                        )

        if detector_scores is not None:
            for i, score in enumerate(detector_scores):
                sns.lineplot(score, ax=axes[n_ts_channels + i], linewidth=1.5)
                axes[n_ts_channels + i].set_ylabel(
                    detector_names[i] if detector_names is not None else f"D{i}"
                )
                axes[n_ts_channels + i].set_ylim(-0.02, 1.02)

        if title is not None:
            fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def get_detectors(self) -> list[str]:
        """Return the sorted list of original anomaly detectors for this kind.

        Returns:
            Sorted list of detector name strings.
        """
        detectors = self.DETECTORS_MULTI if self.kind == "multi" else self.DETECTORS
        return sorted(detectors)

    def get_baselines(self) -> list[str]:
        """Return theoretical benchmarking baselines (Oracle, RandomMS).

        Returns:
            Sorted list of baseline name strings.
        """
        return sorted(self.BASELINES)

    def get_ensembles(self) -> list[str]:
        """Return ensemble methods: hardcoded names plus auto-discovered subdirectories.

        Any subdirectory in scores/<kind>/ that is not a known detector or
        baseline is treated as an ensemble and included in the result.

        Returns:
            Sorted list of ensemble name strings.
        """
        known = set(self.get_detectors()) | set(self.BASELINES)
        discovered = []
        if self.scores_path.exists():
            for name in os.listdir(self.scores_path):
                if (self.scores_path / name).is_dir() and name not in known:
                    discovered.append(name)
        return sorted(set(self.ENSEMBLES) | set(discovered))

    def get_all_models(self) -> list[str]:
        """Return all known models: detectors + ensembles + baselines.

        Returns:
            Sorted list of all model name strings.
        """
        return sorted(set(self.get_detectors()) | set(self.get_ensembles()) | set(self.get_baselines()))

    def load_scores(
        self,
        detectors: Optional[list[str]] = None,
        filenames: Optional[list[str]] = None,
        njobs: Optional[int] = None,
        drop_missing: bool = True,
    ) -> tuple[list[np.ndarray], list[str]]:
        """Load pre-computed anomaly scores for a set of detectors and files.

        Args:
            detectors:    Detectors to load. Defaults to get_detectors().
            filenames:    Time series files to load scores for (CSV names).
                          Defaults to all files in raw/<kind>/.
            njobs:        Number of parallel worker processes. None = cpu_count.
            drop_missing: If True, drop any time series that is missing scores
                          for at least one detector. If False, include partial
                          entries but still drop NaN-containing scores.

        Returns:
            Tuple of (grouped_scores, problematic_files).
            grouped_scores: list of 2-D arrays of shape (n_detectors, T), one
                            per time series.
            problematic_files: list of .npy filenames that were skipped.
        """
        grouped_scores = []
        problematic_files = []

        if not detectors:
            detectors = self.get_detectors()
        if not filenames:
            filenames = self.get_filenames()

        filenames = [name.replace(".csv", ".npy") for name in filenames]
        paths = [
            str(self.scores_path / detector / name)
            for detector in detectors
            for name in filenames
        ]
        with Pool(processes=njobs) as pool:
            scores = list(tqdm(pool.imap(self._load_single_npy, paths), total=len(paths), desc="Loading scores"))

        df_scores = pd.DataFrame(list(zip(paths, scores)), columns=["path", "score"])
        df_scores["filename"] = df_scores["path"].str.split("/").str[-1]
        df_scores["detector"] = df_scores["path"].str.split("/").str[-2]

        none_rows = df_scores[df_scores["score"].isna()]
        problematic_files = list(none_rows.filename)
        if drop_missing:
            df_scores = df_scores[~df_scores.filename.isin(problematic_files)]
        else:
            valid_idx = [i for i in df_scores.index if i not in none_rows.index]
            df_scores = df_scores.loc[valid_idx]

        for name, group in df_scores.groupby("filename"):
            curr_group = list(group.score)
            equal_lengths = np.all([x.shape == curr_group[0].shape for x in curr_group])
            if not equal_lengths:
                problematic_files.append(name)
                continue
            if any(np.isnan(score).any() for score in curr_group):
                problematic_files.append(name)
                continue
            grouped_scores.append(np.stack(list(group.score)))

        return grouped_scores, problematic_files

    def load_runtime(self, models: Optional[list[str]] = None) -> pd.DataFrame:
        """Load runtime CSVs and concatenate them into a single DataFrame.

        Args:
            models: Models to load runtime for. Defaults to get_detectors().
                    Pass get_all_models() to include ensembles and baselines.

        Returns:
            Concatenated DataFrame, or an empty DataFrame if no files are found.
        """
        if models is None:
            models = self.get_detectors()
        dfs = []
        for model in models:
            path = self.runtime_path / f"{model}.csv"
            try:
                dfs.append(pd.read_csv(path))
            except Exception:
                continue
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def load_splits(self) -> dict:
        """Load the official train/eval/label split file lists.

        Returns:
            Dict with keys 'eval' and 'label', each containing a list of
            filename strings belonging to that split.
        """
        prefix = "TSB-AD-" + ("U" if self.kind == "uni" else "M")
        splits = {
            "eval": list(pd.read_csv(self.splits_path / f"{prefix}-Eval.csv")["file_name"]),
            "label": list(pd.read_csv(self.splits_path / f"{prefix}-Label.csv")["file_name"]),
        }
        return splits

    def delete_indexes_from_lists(
        self,
        *lists: list,
        indexes: list[int],
        in_place: bool = True,
    ) -> Optional[list[list]]:
        """Delete elements at given indexes from one or more parallel lists.

        Args:
            *lists:   Two or more lists of equal length to delete from.
            indexes:  Integer positions to delete (duplicates are ignored).
            in_place: If True (default), mutate the lists directly and return
                      None. If False, return new filtered lists without
                      modifying the originals.

        Returns:
            None when in_place=True. A list of filtered lists when
            in_place=False.
        """
        if len(lists) == 0:
            raise ValueError("At least one list must be provided.")

        lengths = [len(curr_list) for curr_list in lists]
        if len(set(lengths)) != 1:
            raise ValueError("All lists must have the same length.")

        if not indexes:
            return list(lists) if not in_place else None

        list_length = lengths[0]
        unique_indexes = sorted(set(indexes), reverse=True)

        for idx in unique_indexes:
            if idx < 0 or idx >= list_length:
                raise IndexError(f"Index out of range: {idx}")

        if in_place:
            for idx in unique_indexes:
                for curr_list in lists:
                    del curr_list[idx]
            return None

        kept_values_mask = np.ones(list_length, dtype=bool)
        kept_values_mask[unique_indexes] = False
        return [[value for i, value in enumerate(curr_list) if kept_values_mask[i]] for curr_list in lists]

    def load_scores_and_delete(
        self,
        fnames: list[str],
        timeseries: list[np.ndarray],
        labels: list[np.ndarray],
        detectors: Optional[list[str]] = None,
        njobs: Optional[int] = None,
    ) -> tuple[list[np.ndarray], list[str]]:
        """Load scores and remove problematic entries from all three input lists.

        Calls load_scores with drop_missing=True, then removes the corresponding
        entries from fnames, timeseries, and labels **in-place** so all three
        lists stay aligned with the returned scores. This avoids copying large
        arrays while keeping the caller's lists consistent.

        Args:
            fnames:     List of CSV filenames (mutated in-place).
            timeseries: List of time series arrays (mutated in-place).
            labels:     List of label arrays (mutated in-place).
            detectors:  Detectors to load. Defaults to get_detectors().
            njobs:      Number of parallel worker processes. None = cpu_count.

        Returns:
            Tuple of (scores, problematic_files) — same as load_scores.
        """
        scores, problematic_files = self.load_scores(
            detectors=detectors,
            filenames=fnames,
            njobs=njobs,
            drop_missing=True,
        )

        if len(problematic_files):
            problematic_csv = [name.replace(".npy", ".csv") for name in problematic_files]
            indexes = [idx for idx, name in enumerate(fnames) if name in problematic_csv]
            self.delete_indexes_from_lists(fnames, timeseries, labels, indexes=indexes)

        assert len(fnames) == len(timeseries) == len(labels) == len(scores), "Lengths not equal!"
        assert all(
            t.shape[0] == l.shape[0] == s.shape[1]
            for t, l, s in zip(timeseries, labels, scores)
        ), "Shape mismatch between time series, labels, and scores!"
        return scores, problematic_files

    def load_metrics(self, models: Optional[list[str]] = None) -> pd.DataFrame:
        """Load per-model evaluation metric CSVs and concatenate them.

        Args:
            models: Models to load metrics for. Defaults to get_detectors().
                    Pass get_all_models() to include ensembles and baselines.

        Returns:
            Concatenated DataFrame with a column for each metric, or an empty
            DataFrame if no files are found.
        """
        if models is None:
            models = self.get_detectors()
        dfs = []
        for model in tqdm(models, desc="Loading metrics"):
            try:
                curr_df = pd.read_csv(self.metrics_path / f"{model}.csv")
                curr_df = self._normalize_metrics_columns(curr_df)
                dfs.append(curr_df)
            except Exception:
                continue
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def _normalize_metrics_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        column_aliases = {
            "VUS-PR": "VUS-PR (L: 128)",
            "VUS-PR Time cost": "VUS-PR (L: 128) Time cost",
        }
        for canonical_name, aliased_name in column_aliases.items():
            if canonical_name not in df.columns and aliased_name in df.columns:
                df[canonical_name] = df[aliased_name]
            if aliased_name not in df.columns and canonical_name in df.columns:
                df[aliased_name] = df[canonical_name]
        return df

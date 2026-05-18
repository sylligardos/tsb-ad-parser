# TSB-AD

Python parser for the **TSB-AD** benchmark (Time Series Benchmark for Anomaly Detection), a large-scale evaluation suite for time series anomaly detection published at NeurIPS 2024.

## What is TSB-AD?

TSB-AD ships with:

- **Raw time series** with ground-truth anomaly labels
- **Pre-computed anomaly scores** for 30+ detectors
- **Evaluation metrics** (VUS-PR, AUC-ROC, etc.) per detector and series
- **Runtime** measurements per detector
- **Official train/eval/label splits**

Both **univariate** (`uni`) and **multivariate** (`multi`) variants are included.

See the [official TSB-AD repository](https://github.com/thedatumorg/TSB-AD) for the full benchmark description and data downloads.

## Repository layout

```
TSB-AD/
├── data/
│   ├── raw/uni|multi/        # Raw time series CSVs (values + label column)  [gitignored]
│   ├── scores/uni|multi/     # Per-detector .npy score files                 [gitignored]
│   ├── metrics/uni|multi/    # Per-detector metric CSVs                      [gitignored]
│   ├── runtime/uni|multi/    # Per-detector runtime CSVs                     [gitignored]
│   └── splits/               # Official TSB-AD-U / TSB-AD-M split CSVs
├── src/tsbadparser/          # The parser package
├── tests/                    # Parser unit tests
├── environment.yml           # Conda environment
└── pyproject.toml            # Makes tsbadparser pip-installable
```

The bulk of the data (raw/, scores/, metrics/, runtime/) is gitignored because of its size. Only the split CSVs are tracked.

## Setup

### 1. Clone this repo

```bash
git clone https://github.com/sylligardos/tsb_ad_parser.git
cd tsb_ad_parser
```

### 2. Download the benchmark data

Download the data from the [official TSB-AD repository](https://github.com/thedatumorg/TSB-AD) and place the files under `data/` following the structure above.

### 3. Create the environment

```bash
conda env create -f environment.yml
conda activate tsbad
```

This installs `tsbadparser` as an editable package. `from tsbadparser import TSBADParser` is available in any script run under the `tsbad` environment, without any extra install step.

## Usage

```python
from tsbadparser import TSBADParser

# Looks for data/ next to this repo by default
parser = TSBADParser(kind="uni")   # or kind="multi"

# Metadata
print(parser.get_datasets())    # ['NAB', 'YAHOO', ...]
print(parser.get_detectors())   # ['AnomalyTransformer', 'IForest', ...]

# Load time series
fnames, timeseries, labels = parser.load_timeseries(dataset=["NAB"])

# Load anomaly scores
scores, bad = parser.load_scores(detectors=["IForest", "LOF"], filenames=fnames)
# scores[i] has shape (n_detectors, T)

# Load scores and drop problematic files (mutates fnames/timeseries/labels in-place)
fnames, timeseries, labels = parser.load_timeseries(dataset=["NAB"])
scores, bad = parser.load_scores_and_delete(fnames, timeseries, labels)

# Metrics and runtime
df_metrics = parser.load_metrics()
df_runtime = parser.load_runtime()

# Official splits
splits = parser.load_splits()   # {'eval': [...], 'label': [...]}

# Visualize
parser.visualize_timeseries(
    timeseries[0],
    anomaly_labels=labels[0],
    title=fnames[0],
    detector_scores=[scores[0][0], scores[0][1]],
    detector_names=["IForest", "LOF"],
)
```

## Benchmark citation

```bibtex
@article{liu2024elephant,
  title     = {The elephant in the room: Towards a reliable time-series anomaly detection benchmark},
  author    = {Liu, Qinghua and Paparrizos, John},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {37},
  pages     = {108231--108261},
  year      = {2024}
}
```

## Credits

- **Benchmark data**: Qinghua Liu & John Paparrizos (NeurIPS 2024)
- **This parser**: Emmanouil Sylligardos, 3rd year PhD, Paris, 2026

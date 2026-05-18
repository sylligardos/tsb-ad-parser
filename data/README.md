# data/

Benchmark data for TSB-AD. The large files are gitignored — only `splits/` and
the directory skeleton are tracked.

## Expected structure

```
data/
├── raw/
│   ├── uni/    ← univariate CSVs  (values columns + binary label column)
│   └── multi/  ← multivariate CSVs
├── scores/
│   ├── uni/    ← one subdir per detector, each containing .npy score files
│   └── multi/
├── metrics/
│   ├── uni/    ← one CSV per detector with evaluation metrics
│   └── multi/
├── runtime/
│   ├── uni/    ← one CSV per detector with wall-clock runtimes
│   └── multi/
└── splits/     ← official TSB-AD-U/M Eval and Label split CSVs (tracked)
```

## Download

The full benchmark is published alongside the NeurIPS 2024 paper:

> Liu, Q. & Paparrizos, J. — *The elephant in the room: Towards a reliable
> time-series anomaly detection benchmark*, NeurIPS 2024.

Download the data from the official repository and place the files in the
subdirectories above, preserving this layout. The parser will find everything
automatically once the structure matches.

## Custom data path

If you store the data elsewhere, pass the path explicitly:

```python
from tsbadparser import TSBADParser
parser = TSBADParser("/path/to/your/data", kind="uni")
```

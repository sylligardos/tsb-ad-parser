import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils import build_anomaly_table, THRESHOLDS, BUFFERS
from tsbadparser import TSBADParser


def main():
    ap = argparse.ArgumentParser(description='Precompute anomaly features and save as parquet.')
    ap.add_argument('--output', default=None,
                    help='Output parquet path (default: data/anomaly_features_{kind}.parquet)')
    ap.add_argument('--kind', default='uni', choices=['uni', 'multi'],
                    help='Benchmark kind (default: uni)')
    args = ap.parse_args()

    output_path = Path(args.output) if args.output else Path(f'data/anomaly_features_{args.kind}.parquet')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Initialising TSBADParser (kind={args.kind}) ...")
    parser = TSBADParser(path='data', kind=args.kind)

    print("Loading all time series ...")
    fnames, timeseries, labels = parser.load_timeseries(load_all=True)
    print(f"  Loaded {len(fnames)} series.")

    detectors = parser.get_detectors()
    print(f"Loading scores for {len(detectors)} detectors ...")
    scores, problematic = parser.load_scores_and_delete(fnames, timeseries, labels, detectors=detectors)
    if problematic:
        print(f"  Dropped {len(problematic)} problematic files.")
    print(f"  {len(fnames)} series retained after score filtering.")

    print("Building anomaly feature table ...")
    df = build_anomaly_table(parser, fnames, timeseries, labels, scores, detectors)

    df.to_parquet(output_path, index=False)

    n_anomalies = len(df)
    n_files = df['filename'].nunique() if n_anomalies > 0 else 0
    print(f"\nDone.")
    print(f"  Total anomaly segments : {n_anomalies}")
    print(f"  Total files            : {n_files}")
    print(f"  Output                 : {output_path}")


if __name__ == '__main__':
    main()

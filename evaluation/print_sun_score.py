import argparse
import pandas as pd
from utils.result_scores import calculate_bootstrap_sun_metrics


def main(args):

    df = pd.read_csv(args.input_csv)
    # Determine model type: struct if scrmsd column exists, else seq
    is_struct = 'scrmsd' in df.columns

    if not is_struct:
        df['plddt'] = df['confidence_score']

    metrics = calculate_bootstrap_sun_metrics(df, is_struct, verbose=False)
    print(f"Success rate (%): {metrics['success_rate_mean']:.1f}±{metrics['success_rate_std']:.1f}")
    print(f"Novel rate (%): {metrics['novel_rate_mean']:.1f}±{metrics['novel_rate_std']:.1f}")
    print(f"Unique rate (%): {metrics['unique_rate']:.1f}")
    print(f"Success unique rate (%): {metrics['success_unique_mean']:.1f}±{metrics['success_unique_std']:.1f}")
    print(f"SUN metric (%): {metrics['sun_metric_mean']:.1f}±{metrics['sun_metric_std']:.1f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print SUN score with components for a model from a result CSV.')
    parser.add_argument('--input-csv', required=True, help='Input CSV file with metrics')
    args = parser.parse_args()
    main(args)
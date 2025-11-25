import argparse
import pandas as pd
import numpy as np
from utils.result_scores import calculate_bootstrap_sun_metrics


def main(args):

    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)

    # Determine model type: struct if scrmsd column exists, else seq
    is_struct = 'scrmsd' in df.columns

    # if not is_struct:
    #     df['plddt'] = df['confidence_score']

    exps = ['single', 'paired']
    both_merics = {
        'sucess': [],
        'sucess novel': [],
        'sucess unique': [],
        'sun': [],
    }
    for exp in exps:
        exp_entries = meta.loc[meta.experiment == exp, 'entry']

        exp_df = df[df.entry.isin(exp_entries)]

        metrics = calculate_bootstrap_sun_metrics(exp_df, is_struct, verbose=False)
        print(f'***{exp.upper()}:')
        print(f"Success rate (%): {metrics['success_rate_mean']:.1f}±{metrics['success_rate_std']:.1f}")
        print(f"Success Novel rate (%): {metrics['novel_rate_mean']:.1f}±{metrics['novel_rate_std']:.1f}")
        print(f"Success Unique rate (%): {metrics['success_unique_mean']:.1f}±{metrics['success_unique_std']:.1f}")
        print(f"SUN metric (%): {metrics['sun_metric_mean']:.1f}±{metrics['sun_metric_std']:.1f}")

        both_merics['sucess'].append(metrics['success_rate_mean'])
        both_merics['sucess novel'].append(metrics['novel_rate_mean'])
        both_merics['sucess unique'].append(metrics['success_unique_mean'])
        both_merics['sun'].append(metrics['sun_metric_mean'])
    
    print('***FINAL:')
    print(f"Success rate (%): {np.mean(both_merics['sucess']):.1f}")
    print(f"Success Novel rate (%): {np.mean(both_merics['sucess novel']):.1f}")
    print(f"Success Unique rate (%): {np.mean(both_merics['sucess unique']):.1f}")
    print(f"SUN metric (%): {np.mean(both_merics['sun']):.1f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print SUN score with components for a model from a result CSV.')
    parser.add_argument('--input-csv', required=True, help='Input CSV file with metrics')
    parser.add_argument('--meta-csv', required=True, help='CSV file with metadata')
    args = parser.parse_args()
    main(args)
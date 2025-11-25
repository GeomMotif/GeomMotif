import pandas as pd
import numpy as np


SUN_TM_THRESHOLD = 0.8  # TM-score threshold for novelty


def get_sr(res_table, is_struct, verbose=True):
    if is_struct:
        quality_df = res_table[res_table['scrmsd'] <= 2].groupby('entry')
        geom_df = res_table[res_table['rmsd'] <= 1].groupby('entry')
        success_df = res_table[(res_table['scrmsd'] <= 2) & (res_table['rmsd'] <= 1)]
    else:
        quality_df = res_table[res_table['plddt'] >= 70].groupby('entry')
        geom_df = res_table[res_table['rmsd'] <= 1].groupby('entry')
        success_df = res_table[(res_table['plddt'] >= 70) & (res_table['rmsd'] <= 1)]

    quality_rates = quality_df.size() / res_table.groupby('entry').size()
    geom_rates = geom_df.size() / res_table.groupby('entry').size()
    success_rates = success_df.groupby('entry').size() / res_table.groupby('entry').size()

    quality_rates = quality_rates.fillna(0)
    geom_rates = geom_rates.fillna(0)
    success_rates = success_rates.fillna(0)

    if verbose:
        print(f"Quality passed (%): {quality_rates.mean() * 100:.2f}")
        print(f"Geometry passed (%): {geom_rates.mean() * 100:.2f}")
        print(f"raw SR (Success rate (%)): {success_rates.mean() * 100:.2f}")

    return success_df


def _per_entry_rates(df: pd.DataFrame, is_struct: bool) -> tuple[pd.Series, pd.Series]:
    """Compute per-entry success and novel rates for a given dataframe."""
    success_df = get_sr(df, is_struct, verbose=False)
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    by_entry_counts = df.groupby('entry').size()
    if by_entry_counts.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Success rate per entry
    success_rates = success_df.groupby('entry').size() / by_entry_counts
    success_rates = success_rates.reindex(by_entry_counts.index).fillna(0.0)

    # Novel among successes per entry
    novel_success = success_df[success_df['struct_novelty_tmscore'] <= SUN_TM_THRESHOLD]
    novel_rates = novel_success.groupby('entry').size() / by_entry_counts
    novel_rates = novel_rates.reindex(by_entry_counts.index).fillna(0.0)

    return success_rates, novel_rates


def calculate_bootstrap_sun_metrics(full_df: pd.DataFrame,
                                    is_struct: bool,
                                    n_bootstrap_iterations: int = 5,
                                    bootstrap_sample_size: int = 100,
                                    verbose: bool = True,
                                    random_state: int | np.random.Generator = 42) -> dict:
    """
    Calculate Success, Unique, Novel and SUN metrics using bootstrap.

    - Unique rate is computed once on the full novel-success set as clustering density.
    - For each bootstrap iteration, sample per-entry with replacement and compute:
      success rate, novel_success rate (novel among total), and sun_metric per-entry as
      novel_success_rate * predefined_unique_rate.

    Returns dict with means and stds (where applicable), all in percent.
    """
    rng = np.random.default_rng(random_state)
    # Precompute unique rate from the full table (clustering density among novel successes)
    success_df_full = get_sr(full_df, is_struct, verbose=False)
    # print(success_df_full)
    if success_df_full.empty:
        if verbose:
            print('No successes found. Metrics default to 0.')
        return {
            'success_rate_mean': 0.0, 'success_rate_std': 0.0,
            'novel_rate_mean': 0.0, 'novel_rate_std': 0.0,
            'unique_rate': 0.0,
            'sun_metric_mean': 0.0, 'sun_metric_std': 0.0,
        }

    novel_success_full = success_df_full[success_df_full['struct_novelty_tmscore'] <= SUN_TM_THRESHOLD]

    # relative_unique is computed on the full data and reused in bootstrap
    if not novel_success_full.empty:
        relative_unique = (
            novel_success_full.groupby('entry')['struct_cluster'].nunique()
            / novel_success_full.groupby('entry').size()
        )
        unique_rate = float(relative_unique.fillna(0.0).mean()) * 100.0
    else:
        relative_unique = pd.Series(dtype=float)
        unique_rate = 0.0

    # Bootstrap success and novel rates and SUN metric
    success_boot = []
    novel_boot = []
    success_unique_boot = []
    sun_boot = []

    grouped = full_df.groupby('entry')
    for _ in range(n_bootstrap_iterations):
        # Sample with replacement within each entry group
        iteration_seed = int(rng.integers(np.iinfo(np.int32).max, dtype=np.int64))
        sampled_df = grouped.sample(bootstrap_sample_size, replace=True, random_state=iteration_seed)

        success_rates, novel_rates = _per_entry_rates(sampled_df, is_struct)

        # Align with relative_unique index when computing SUN
        if not relative_unique.empty:
            aligned_unique = relative_unique.reindex(success_rates.index).fillna(0.0)
        else:
            aligned_unique = pd.Series(0.0, index=success_rates.index)

        # Aggregate metrics across entries
        success_boot.append(float(success_rates.mean()) * 100.0)
        novel_boot.append(float(novel_rates.mean()) * 100.0)

        success_unique = success_rates * aligned_unique
        success_unique_boot.append(float(success_unique.mean()) * 100.0)

        # SUN per-entry: intersection rate (novel & success over total) * predefined unique
        sun_value = (novel_rates * aligned_unique).mean() * 100.0
        sun_boot.append(float(sun_value))

    metrics = {
        'success_rate_mean': float(np.mean(success_boot)),
        'success_rate_std': float(np.std(success_boot, ddof=0)),
        'novel_rate_mean': float(np.mean(novel_boot)),
        'novel_rate_std': float(np.std(novel_boot, ddof=0)),
        'unique_rate': unique_rate,
        'success_unique_mean': float(np.mean(success_unique_boot)),
        'success_unique_std': float(np.std(success_unique_boot, ddof=0)),
        'sun_metric_mean': float(np.mean(sun_boot)),
        'sun_metric_std': float(np.std(sun_boot, ddof=0)),
    }

    if verbose:
        print(f"Success rate (%): {metrics['success_rate_mean']:.2f}±{metrics['success_rate_std']:.2f}")
        print(f"Novel rate (%): {metrics['novel_rate_mean']:.2f}±{metrics['novel_rate_std']:.2f}")
        print(f"Unique rate (%): {metrics['unique_rate']:.2f}")
        print(f"SUN metric (%): {metrics['sun_metric_mean']:.2f}±{metrics['sun_metric_std']:.2f}")

    return metrics
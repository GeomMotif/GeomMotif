import pandas as pd
import numpy as np
from tqdm import tqdm
import subprocess
from scipy.cluster.hierarchy import linkage, fcluster
import os
import concurrent.futures
import argparse


def calculate_coverage(lens_str: str) -> float:
    """Calculate coverage ratio between two structure lengths"""
    len1, len2 = map(float, lens_str.split(':'))
    return min(len1, len2) / max(len1, len2)

def tmalign_pair(args):
    i, j, pdb1, pdb2 = args
    try:
        result = subprocess.run(
            ["TMalign", pdb1, pdb2],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        tm_score = np.nan
        len1 = len2 = None
        for line in result.stdout.splitlines():
            if line.strip().startswith("TM-score="):
                tm_score = float(line.strip().split('=')[1].split()[0])
            if "Length of Chain_1:" in line:
                len1 = int(line.strip().split()[-2])
            if "Length of Chain_2:" in line:
                len2 = int(line.strip().split()[-2])
        coverage = np.nan
        if len1 is not None and len2 is not None:
            coverage = calculate_coverage(f"{len1}:{len2}")
        return (i, j, tm_score, coverage)
    except Exception as e:
        print(f"TM-align failed for {pdb1} vs {pdb2}: {e}")
        return (i, j, np.nan, np.nan)

def run_tmalign_matrix(pdb_files, n_workers=8, min_coverage=0.0):
    """
    Compute the pairwise TM-score matrix for a list of PDB files using TM-align in parallel.
    Returns a symmetric matrix of shape (n, n).
    """
    n = len(pdb_files)
    tm_matrix = np.ones((n, n))  # Diagonal is 1.0

    # Prepare all pairs (i, j) with i < j
    pairs = [(i, j, pdb_files[i], pdb_files[j]) for i in range(n) for j in range(i+1, n)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        for i, j, tm_score, coverage in tqdm(executor.map(tmalign_pair, pairs), total=len(pairs)):
            if coverage >= min_coverage:
                tm_matrix[i, j] = tm_score
                tm_matrix[j, i] = tm_score  # Symmetric
            else:
                tm_matrix[i, j] = 0
                tm_matrix[j, i] = 0


    return tm_matrix

def cluster_structures(tm_matrix, threshold=0.7):
    """
    Cluster structures using complete linkage on 1-TM distance.
    Returns cluster labels for each structure.
    """
    # Convert TM-scores to distances (1 - TM-score)
    distance_matrix = 1 - tm_matrix
    # Use only the upper triangle for condensed distance matrix
    condensed = distance_matrix[np.triu_indices(len(distance_matrix), k=1)]
    # Perform hierarchical clustering with complete linkage
    linkage_matrix = linkage(condensed, method='complete')
    # Assign clusters: threshold is 1-TM, so 0.3 means TM >= 0.7
    clusters = fcluster(linkage_matrix, t=1-threshold, criterion='distance')
    return clusters

def tmalign_single_idx(args):
    idx, pdb_init, pdb_iter = args
    if os.path.isfile(pdb_iter) and os.path.isfile(pdb_init):
        try:
            result = subprocess.run(
                ["TMalign", pdb_init, pdb_iter],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            for line in result.stdout.splitlines():
                if line.strip().startswith("TM-score="):
                    return idx, float(line.strip().split('=')[1].split()[0])
            return idx, np.nan
        except Exception as e:
            print(f"TM-align failed for {pdb_init} vs {pdb_iter}: {e}")
            return idx, np.nan
    else:
        return idx, np.nan
        
def main(args):
    # Use args parameters instead of hardcoded values
    ref_pdbs_dir = args.ref_pdb_dir
    input_dir = args.input_csv
    output_dir = args.output_csv
    pdbs_dir = args.gen_pdb_dir
    n_cors = args.n_cors

    # Load your results DataFrame
    df = pd.read_csv(input_dir)
    df['iteration'] = df['iteration'].astype(int)
    if 'scrmsd' in df.columns:
        success_df = df[(df['rmsd'] <= 1.0) & (df['scrmsd'] <= 2.0)]
    else:
        success_df = df[(df['rmsd'] <= 1.0) & (df['plddt'] >= 70)]

    ########## Structure Diversity: clustering by pairwise TM-score
    df['struct_cluster'] = None
    print('Running structure clustering')

    for entry in tqdm(success_df['entry'].unique()):
        
        entry_mask = success_df['entry'] == entry
        entry_df = success_df[entry_mask]

        pdb_files = []
        for idx, row in entry_df.iterrows():
            pdb_iter = os.path.join(pdbs_dir, f'{row["gen_pdb_name"]}.pdb')
            pdb_files.append(pdb_iter)

        if len(pdb_files) > 1:
            tm_matrix = run_tmalign_matrix(pdb_files, n_workers=n_cors, min_coverage=args.struct_div_cov)
            if np.any(tm_matrix > 0):
                # tm_matrix = np.where((np.isnan(tm_matrix)) | (tm_matrix == 0), 0.00000, tm_matrix)
                clusters = cluster_structures(tm_matrix, threshold=args.struct_div_tm)
                df.loc[entry_df.index, 'struct_cluster'] = clusters
            else:
                df.loc[entry_df.index, 'struct_cluster'] = 0
        else:
            df.loc[entry_df.index, 'struct_cluster'] = 0

    # Save results
    df.to_csv(output_dir, index=False)

    ########## Structural Novelty: TM-score to initial structure
    print('Running structural novelty (TM-score to initial structure) in parallel')
    df['struct_novelty_tmscore'] = np.nan

    # Prepare arguments for parallel execution
    novelty_args = []
    for idx, row in df.iterrows():
        pdb_iter = os.path.join(pdbs_dir, f'{row["gen_pdb_name"]}.pdb')
        pdb_id = row["entry"].split('_')[1]
        pdb_init = os.path.join(ref_pdbs_dir, f'{pdb_id}.pdb')
        novelty_args.append((idx, pdb_init, pdb_iter))

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cors) as executor:
        for idx, tmscore in tqdm(executor.map(tmalign_single_idx, novelty_args), total=len(novelty_args)):
            df.at[idx, 'struct_novelty_tmscore'] = tmscore

    # Save results
    df.to_csv(output_dir, index=False)
    print('Done')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--gen-pdb-dir', type=str, required=True)
    parser.add_argument('--ref-pdb-dir', type=str, required=True)
    
    parser.add_argument('--struct_div_cov', type=float, default=0.9)
    parser.add_argument('--struct_div_tm', type=float, default=0.8)
    
    parser.add_argument('--n_cors', type=int, required=True)

    args = parser.parse_args()

    main(args)
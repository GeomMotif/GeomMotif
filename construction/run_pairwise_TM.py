import argparse
import os
import numpy as np
from typing import *
from tqdm import tqdm
import multiprocessing as mp

from utils.tmalign_runner import run_tmalign

def calculate_coverage(lens_str: str) -> float:
    """Calculate coverage ratio between two structure lengths"""
    len1, len2 = map(float, lens_str.split(':'))
    return min(len1, len2) / max(len1, len2)

def process_query(args: Tuple[str, str, bool]) -> Tuple[float, str]:
    """Process a single query against all references in parallel"""
    query, ref, fast = args
    return run_tmalign(query, ref, fast)

def parallel_process_folder(query_pdbs: str, reference_pdbs: List[str], n_workers: int = 4) -> Tuple[List[float], List[str]]:
    """Process all queries in a folder in parallel"""
    with mp.Pool(n_workers) as pool:
        args = [(query_pdb, ref, True) 
               for i, query_pdb in enumerate(query_pdbs) 
               for j, ref in enumerate(reference_pdbs) 
               if i < j]

        results = list(tqdm(pool.imap(process_query, args), total=len(args)))

    return results

def get_pdb_files(pdb_dir, pdb_names= None):
    if pdb_names:
        return [os.path.join(pdb_dir, pdb_file) for pdb_file in pdb_names]
    else:
        return [os.path.join(pdb_dir, pdb_file) for pdb_file in os.listdir(pdb_dir) if '.pdb' in pdb_file]
        
def create_pairwise_matrix(results, pdb_files, min_coverage: float = 0.0):
    """Create a pairwise matrix of TM scores and a list of PDB names"""
    pdb_names = [os.path.basename(pdb) for pdb in pdb_files]
    n = len(pdb_names)
    
    tm_matrix = np.full((n, n), np.nan)
    
    np.fill_diagonal(tm_matrix, 1.0)
    
    name_to_idx = {name: i for i, name in enumerate(pdb_names)}
    
    for result in results:
        if len(result) >= 4:
            tm_score, lens_str, query_path, ref_path = result
            coverage = calculate_coverage(lens_str)
            
            # Skip if coverage is below threshold
            if coverage < min_coverage:
                continue
                
            query_name = os.path.basename(query_path)
            ref_name = os.path.basename(ref_path)
            
            if query_name in name_to_idx and ref_name in name_to_idx:
                i = name_to_idx[query_name]
                j = name_to_idx[ref_name]
                # since TM-align scores are symmetric:
                tm_matrix[i, j] = tm_score
                tm_matrix[j, i] = tm_score
    
    return tm_matrix, pdb_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_pdbs_dir")
    parser.add_argument("output_csv")
    parser.add_argument("--n_workers", default=1, type=int)
    parser.add_argument("--matrix_output", default=None, help="Path to save the pairwise TM score matrix")
    parser.add_argument("--min_coverage", type=float, default=0.0, 
                      help="Minimum coverage ratio between structure lengths (0.0-1.0)")
    args = parser.parse_args()

    print(f'''
    **Calculating TM-align statistics**:
    Query dataset: {args.input_pdbs_dir}
    ''')

    query_pdbs = get_pdb_files(args.input_pdbs_dir)

    results= []
    
    results = parallel_process_folder(query_pdbs, query_pdbs, int(args.n_workers))
        
    with open(args.output_csv, 'w') as f:
        for result in results:
            coverage = calculate_coverage(result[1])
            f.write(f'{result[2]}|{result[3]}|{result[0]}|{coverage}\n')
    
    matrix_output = args.matrix_output or f"{os.path.splitext(args.output_csv)[0]}_matrix.npz"
    tm_matrix, pdb_names = create_pairwise_matrix(results, query_pdbs, args.min_coverage)
    
 
    np.savez(matrix_output, 
             tm_matrix=tm_matrix, 
             pdb_names=np.array(pdb_names, dtype=str))
    
    print(f"Saved pairwise TM score matrix to {matrix_output}")
    
    # Calculate and print average pairwise TM score
    valid_scores = tm_matrix[~np.isnan(tm_matrix) & (tm_matrix != 1.0)]  # Exclude diagonal and NaN values
    if len(valid_scores) > 0:
        avg_tm_score = np.mean(valid_scores)
        print(f'Average pairwise TM score: {round(avg_tm_score, 3)}')
        print(f'Number of pairs above coverage threshold: {len(valid_scores)}')

'''
python run_pairwise_TM.py \
    ./test_pdbs_from_mutated_fasta_substructures_united_filtered_esmfold_dist10 \
    dist10_pairwise_TM.csv \
    test \
    --n_workers 14 \
    --min_coverage 0.8
'''

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
import argparse
from pathlib import Path
import sys


def load_tm_matrix(matrix_file):
    data = np.load(matrix_file)
    matrix = data['tm_matrix']
    matrix = np.where((np.isnan(matrix)) | (matrix == 0), 0.00000, matrix)
    return matrix, data['pdb_names']


def analyze_clusters(tm_matrix, cutoffs, method='average'):
    distance_matrix = 1 - tm_matrix
    
    linkage_matrix = linkage(distance_matrix[np.triu_indices(len(distance_matrix), k=1)], 
                           method=method)
    
    results = {}
    for cutoff in cutoffs:
        distance_cutoff = 1 - cutoff
        
        clusters = fcluster(linkage_matrix, distance_cutoff, criterion='distance')
        n_clusters = len(np.unique(clusters))
        cluster_sizes = np.bincount(clusters)
        
        results[cutoff] = {
            'n_clusters': n_clusters,
            'cluster_labels': clusters,
            'cluster_sizes': cluster_sizes
        }
    
    return results, linkage_matrix


def main():
    parser = argparse.ArgumentParser(description='Analyze TM-score clustering at different cutoffs')
    parser.add_argument('matrix_file', help='NPZ file containing TM-score matrix')
    parser.add_argument('--output_prefix', default='cluster_analysis',
                       help='Prefix for output files')
    parser.add_argument('--methods', nargs='+', 
                       default=['single', 'complete', 'average', 'ward'],
                       help='Linkage methods to compare')
    args = parser.parse_args()

    tm_matrix, pdb_names = load_tm_matrix(args.matrix_file)
    
    cutoffs = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    output_dir = Path(args.output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    linkage_matrices = {}
    for method in args.methods:
        try:
            results, linkage_matrix = analyze_clusters(tm_matrix, cutoffs, method)
            all_results[method] = results
            linkage_matrices[method] = linkage_matrix
            
        except Exception as e:
            print(f"Error processing method {method}: {str(e)}")
            continue
    
    if not all_results:
        print("No clustering methods were successful")
        sys.exit(1)
    
    for method in args.methods:
        if method not in all_results:
            continue
        print(f"\nClustering Results ({method} linkage):")
        print("-" * 40)
        print("TM-score cutoff | Number of clusters")
        print("-" * 40)
        for cutoff in cutoffs:
            print(f"{cutoff:13.1f} | {all_results[method][cutoff]['n_clusters']:17d}")
    
    for method in args.methods:
        if method not in all_results:
            continue
        with open(f'{args.output_prefix}_{method}_assignments.txt', 'w') as f:
            f.write("Structure" + "\t" + "Cluster assignments at different cutoffs\n")
            header = "Name" + "\t" + "\t".join(f"TM>{c}" for c in cutoffs) + "\n"
            f.write(header)
            for i, name in enumerate(pdb_names):
                clusters = [all_results[method][c]['cluster_labels'][i] for c in cutoffs]
                line = name + "\t" + "\t".join(map(str, clusters)) + "\n"
                f.write(line)


if __name__ == '__main__':
    main()
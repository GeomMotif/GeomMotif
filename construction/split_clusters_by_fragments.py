import argparse
import os
from pathlib import Path
import numpy as np
from collections import defaultdict

def get_fragments(indices_str):
    """Convert string of indices into fragments"""
    indices = [int(x) for x in indices_str.split('_')]
    sorted_indices = sorted(indices)
    
    fragments = []
    current_fragment = [sorted_indices[0]]
    
    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] == sorted_indices[i-1] + 1:
            # Consecutive residue, add to current fragment
            current_fragment.append(sorted_indices[i])
        else:
            # Gap found, start a new fragment
            fragments.append(current_fragment)
            current_fragment = [sorted_indices[i]]
    
    # Add the last fragment
    fragments.append(current_fragment)
    
    return fragments

def get_indices_set(indices_str):
    """Convert indices string to set of integers"""
    return set(int(x) for x in indices_str.split('_'))

def calculate_overlap(indices1, indices2):
    """Calculate overlap percentage between two sets of indices"""
    set1 = set(indices1)
    set2 = set(indices2)
    intersection = len(set1.intersection(set2))
    smaller_set = min(len(set1), len(set2))
    return intersection / smaller_set if smaller_set > 0 else 0

def filter_duplicates(structures):
    """Filter duplicates based on name prefix and index overlap"""
    # Group structures by their name prefix
    prefix_groups = defaultdict(list)
    for struct in structures:
        prefix = struct['name'].split('_')[0]
        prefix_groups[prefix].append(struct)
    
    # For each group, filter duplicates
    filtered_structures = []
    for prefix, group in prefix_groups.items():
        # Sort by name to ensure consistent selection
        group.sort(key=lambda x: x['name'])
        
        # Keep track of which structures to remove
        to_remove = set()
        
        # Compare each pair of structures
        for i in range(len(group)):
            if i in to_remove:
                continue
            indices1 = get_indices_set(group[i]['indices'])
            
            for j in range(i + 1, len(group)):
                if j in to_remove:
                    continue
                indices2 = get_indices_set(group[j]['indices'])
                
                # Calculate overlap
                overlap = calculate_overlap(indices1, indices2)
                
                # If overlap > 50%, mark the second structure for removal
                if overlap > 0.5:
                    to_remove.add(j)
        
        # Add non-removed structures to filtered list
        filtered_structures.extend([struct for i, struct in enumerate(group) if i not in to_remove])
    
    return filtered_structures

def print_cluster_stats(clusters_by_fragments, fragment_counts):
    """Print detailed statistics about clusters for each fragment count"""
    print("\nDetailed Fragment and Cluster Statistics:")
    print("=" * 80)
    print(f"{'#Fragments':<10} {'#Structures':<12} {'#Clusters':<10} {'Cluster Sizes (after filtering)':<48}")
    print("-" * 80)
    
    for num_fragments, count in sorted(fragment_counts.items()):
        clusters = clusters_by_fragments[num_fragments]
        
        # Calculate filtered sizes for each cluster
        filtered_sizes = []
        for cluster_id, structures in clusters.items():
            filtered_structures = filter_duplicates(structures)
            if filtered_structures:  # Only include non-empty clusters
                filtered_sizes.append(len(filtered_structures))
        
        # Remove empty clusters and sort sizes
        filtered_sizes.sort(reverse=True)
        num_clusters = len(filtered_sizes)
        
        # Format cluster sizes as a compact string
        if len(filtered_sizes) > 11:
            sizes_str = f"[{', '.join(map(str, filtered_sizes[:10]))}...{filtered_sizes[-1]}]"
        else:
            sizes_str = str(filtered_sizes)
        
        print(f"{num_fragments:<10} {count:<12} {num_clusters:<10} {sizes_str:<48}")
    print("=" * 80)

def write_cluster_table(clusters, output_file, stats):
    """Write clusters to a two-column table format"""
    with open(output_file, 'w') as f:
        for cluster_id, structures in sorted(clusters.items()):
            # Filter duplicates within each cluster
            filtered_structures = filter_duplicates(structures)
            if filtered_structures:  # Only write non-empty clusters
                for struct in filtered_structures:
                    f.write(f"{struct['name']}\t{cluster_id}\n")

def main():
    parser = argparse.ArgumentParser(description='Split clusters based on number of fragments')
    parser.add_argument('cluster_assignments', help='Cluster assignments file from analyze_tm_clusters.py')
    parser.add_argument('--output_dir', default='fragment_clusters', help='Output directory')
    parser.add_argument('--tm_cutoff', type=float, default=0.7, help='TM-score cutoff to use')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Read cluster assignments
    with open(args.cluster_assignments, 'r') as f:
        lines = f.readlines()
        header = lines[1].strip().split('\t')
        tm_cutoffs = [float(x.replace('TM>', '')) for x in header[1:]]
        
        # Find the column index for the desired TM cutoff
        try:
            tm_col_idx = tm_cutoffs.index(args.tm_cutoff) + 1
        except ValueError:
            print(f"Error: TM cutoff {args.tm_cutoff} not found in file. Available cutoffs: {tm_cutoffs}")
            return

    # Group structures by number of fragments
    clusters_by_fragments = defaultdict(lambda: defaultdict(list))
    fragment_counts = defaultdict(int)

    for line in lines[2:]:  # Skip header rows
        parts = line.strip().split('\t')
        name = parts[0]
        cluster = int(parts[tm_col_idx])
        
        # Extract indices from filename (after the third underscore)
        try:
            indices_str = '_'.join(name.split('_')[3:]).replace('.pdb', '')
            fragments = get_fragments(indices_str)
            num_fragments = len(fragments)
            
            # Store structure in appropriate cluster and fragment group
            clusters_by_fragments[num_fragments][cluster].append({
                'name': name,
                'fragments': fragments,
                'indices': indices_str
            })
            fragment_counts[num_fragments] += 1
            
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue

    # Write separate files for each number of fragments
    total_before = sum(fragment_counts.values())
    filtered_counts = defaultdict(int)
    
    for num_fragments, clusters in clusters_by_fragments.items():
        output_file = os.path.join(args.output_dir, f'clusters_{num_fragments}_fragments.tsv')
        
        # Count structures after filtering and prepare filtered clusters
        filtered_total = 0
        filtered_clusters = {}
        for cluster_id, structures in clusters.items():
            filtered_structures = filter_duplicates(structures)
            if filtered_structures:  # Only keep non-empty clusters
                filtered_clusters[cluster_id] = filtered_structures
                filtered_total += len(filtered_structures)
        
        filtered_counts[num_fragments] = filtered_total
        
        if filtered_total > 0:  # Only write file if there are structures after filtering
            # Write the table
            write_cluster_table(clusters, output_file, None)

    # Print detailed statistics
    print("\nFiltering Summary:")
    print(f"Total structures before filtering: {total_before}")
    print(f"Total structures after filtering: {sum(filtered_counts.values())}")
    print(f"Removed duplicates: {total_before - sum(filtered_counts.values())}")
    
    print_cluster_stats(clusters_by_fragments, filtered_counts)

if __name__ == '__main__':
    main()

# Example usage:
# python split_clusters_by_fragments.py clustering_results/cluster_analysis_dist10_cov08_complete_assignments.txt --output_dir clustering_results/fragment_clusters_dist10_cov08 --tm_cutoff 0.8 
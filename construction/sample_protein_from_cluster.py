import pandas as pd
import random
import argparse

parser = argparse.ArgumentParser(description="Sample one protein from each cluster")
parser.add_argument("assignments_file", help="Path to cluster assignments file")
parser.add_argument("output_file", help="Path to output sampled clusters file")
parser.add_argument("--tm_cutoff", type=float, default=0.5, help="TM-score cutoff for clustering")
args = parser.parse_args()

df_clusters = pd.read_table(args.assignments_file, skiprows=1)
# print(df_clusters.head())

# Get all unique clusters
clusters = df_clusters[f'TM>{args.tm_cutoff}'].unique()
print(len(clusters))
pdbs = []
for cluster in clusters:
    # Get all PDBs in this cluster
    cluster_pdbs = df_clusters[df_clusters[f'TM>{args.tm_cutoff}'] == cluster]['Name'].tolist()
    # Sample 1 random PDB from the cluster
    sampled_pdb = random.choice(cluster_pdbs)
    pdbs.append(sampled_pdb)

print(f"Total pdbs sampled: {len(pdbs)}")

with open(args.output_file, 'w') as f:
    for pdb in pdbs:
        f.write(f'{pdb}\n')

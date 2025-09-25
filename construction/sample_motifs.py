import pandas as pd
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)

# Read paired environments metadata
df = pd.read_csv('paired_environments.csv')

# Helper function to get residue set from residue string
def get_residue_set(residue_str):
    return set(int(x) for x in residue_str.split('_'))

# Helper function to calculate overlap ratio
def calc_overlap_ratio(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union
print(df.columns)
# Process single environments
single_envs = []
for env_col in ['env1_resid', 'env2_resid']:
    # Keep all relevant metadata columns
    env_metadata_cols = ['pdb_id', env_col, 
                        f"{env_col[:4]}_intervals",
                        f"{env_col[:4]}_total_span",
                        f"{env_col[:4]}_fragment_count"]
    env_data = df[env_metadata_cols].drop_duplicates()
    env_data.columns = ['pdb_id', 'res_ids', 'intervals', 'total_len', 'num_fragments']
    single_envs.append(env_data)
print(df.columns)
single_envs_df = pd.concat(single_envs).drop_duplicates()
print(single_envs_df[single_envs_df.num_fragments == 1])

# Remove overlapping single environments
filtered_single = []
for _, row in tqdm(single_envs_df.iterrows(), desc="Filtering single environments"):
    residues = get_residue_set(row.res_ids)
    overlap = False
    
    for existing in filtered_single:
        # Only check overlap if from same PDB
        if existing['pdb_id'] == row.pdb_id and existing['num_fragments'] == row.num_fragments:
            existing_residues = get_residue_set(existing['res_ids'])
            if calc_overlap_ratio(residues, existing_residues) > 0.2:
                overlap = True
                break
            
    if not overlap:
        filtered_single.append(row.to_dict())

# Sample single environments by num_fragments
used_pdbs = set()
single_samples = []
for num_frags in range(1,8):
    envs = [x for x in filtered_single if x['num_fragments'] == num_frags]
    print('for', num_frags, 'envs', len(envs))
    if len(envs) > 5:
        sampled = np.random.choice(len(envs), size=5, replace=False)
        single_samples.extend([envs[i] for i in sampled])
    else:
        single_samples.extend(envs)

# Process paired environments
filtered_paired = []
for _, row in tqdm(df.iterrows(), desc="Filtering paired environments"):
    env1_residues = get_residue_set(row.env1_resid)
    env2_residues = get_residue_set(row.env2_resid)
    current_residues = env1_residues | env2_residues
    
    overlap = False
    for existing in filtered_paired:
        # Only check overlap if from same PDB
        if existing['pdb_id'] == row.pdb_id:
            existing_env1 = get_residue_set(existing['env1_resid'])
            existing_env2 = get_residue_set(existing['env2_resid'])
            existing_residues = existing_env1 | existing_env2
            
            if calc_overlap_ratio(current_residues, existing_residues) > 0.2:
                overlap = True
                break
            
    if not overlap:
        # Store with standardized column names
        paired_row = {
            'pdb_id': row.pdb_id,
            'intervals': row.united_intervals,
            'total_len': row.total_pdb_residues,
            'res_ids': '_'.join(sorted([str(x) for x in current_residues])),
            'num_fragments': row.env1_fragment_count + row.env2_fragment_count,
            'env1_resid': row.env1_resid,
            'env2_resid': row.env2_resid
        }
        filtered_paired.append(paired_row)

# Sample paired environments by total fragments
used_pdbs = set()
paired_samples = []
for total_frags in range(3,8):
    envs = [x for x in filtered_paired if x['num_fragments'] == total_frags]
    envs = [x for x in envs if x['pdb_id'] not in used_pdbs]
    used_pdbs.update(x['pdb_id'] for x in envs)
    
    print('for', total_frags, 'envs', len(envs))
    if len(envs) > 5:    
        sampled = np.random.choice(len(envs), size=5, replace=False)
        paired_samples.extend([envs[i] for i in sampled])
    else:
        paired_samples.extend(envs)

# Save results with standardized columns
single_df = pd.DataFrame(single_samples)[['pdb_id', 'intervals', 'total_len', 'res_ids']]
paired_df = pd.DataFrame(paired_samples)[['pdb_id', 'intervals', 'total_len', 'res_ids', 'env1_resid', 'env2_resid']]

print(f"Sampled {len(single_df)} single motifs")
print(f"Sampled {len(paired_df)} paired motifs")

single_df.to_csv('sampled_single_motifs.csv', index=False)
paired_df.to_csv('sampled_paired_motifs.csv', index=False)

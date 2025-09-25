import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser, PDBIO, Select, DSSP
from collections import defaultdict
from itertools import combinations
import pandas as pd


class SpecificResiduesSelect(Select):
    """Select only specific residues by their chain and residue IDs."""
    def __init__(self, selected_residues):
        # selected_residues should be a set of (chain_id, residue_id) tuples
        self.selected_residues = selected_residues

    def accept_residue(self, residue):
        return (residue.parent.id, residue.id[1]) in self.selected_residues

class SingleNeighborhoodSelect(Select):
    """Select residues within a certain distance of a central residue."""
    def __init__(self, central_residue, distance_threshold=10.0):
        self.central_residue = central_residue
        self.distance_threshold = distance_threshold
        try:
            self.central_ca_coord = central_residue['CA'].get_coord()
        except KeyError:
            self.central_ca_coord = None

    def accept_residue(self, residue):
        if residue.id[0] != ' ' or self.central_ca_coord is None:  # Skip hetero-residues
            return 0
            
        try:
            ca_coord = residue['CA'].get_coord()
            dist = np.sqrt(np.sum((ca_coord - self.central_ca_coord)**2))
            return 1 if dist <= self.distance_threshold else 0
        except KeyError:
            return 0


def save_single_environment(structure, env_info, output_dir, pdb_id):
    """Save a specific single environment as a PDB file."""
    # Get central residue info
    central_res = env_info['central_residue']
    
    # Get all residue IDs in the environment
    all_residues = set()
    for fragment in env_info['fragments']:
        for resid in fragment:
            all_residues.add(resid)
    
    # Create filename
    residue_list = sorted(list(all_residues))
    filename = f"{pdb_id}+{central_res['resid']}+{'_'.join(map(str, residue_list))}.pdb"
    output_path = os.path.join(output_dir, filename)
    
    # Create selector for these specific residues
    selected_residues = set()
    for resid in all_residues:
        selected_residues.add((central_res['chain'], resid))
    
    # Save the selected residues
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path, SpecificResiduesSelect(selected_residues))
    
    return filename

def check_sequence_gaps(residue_indices):
    """Check sequence gaps and return fragment information."""
    sorted_indices = sorted(residue_indices)
    fragments = []
    current_fragment = [sorted_indices[0]]
    
    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] == sorted_indices[i-1] + 1:
            current_fragment.append(sorted_indices[i])
        else:
            fragments.append(current_fragment)
            current_fragment = [sorted_indices[i]]
    fragments.append(current_fragment)
    
    # Count single-residue fragments
    single_residue_fragments = sum(1 for frag in fragments if len(frag) == 1)
    
    return fragments, single_residue_fragments

def get_residue_plddt(residue):
    """Get pLDDT score from residue B-factor field."""
    try:
        return residue['CA'].get_bfactor()
    except:
        return 0

def calculate_ss_for_structure(structure, pdb_path):
    """Calculate secondary structure for all residues in the structure using DSSP."""
    ss_info = defaultdict(dict)
    
    try:
        # Run DSSP on the structure
        model = structure[0]
        dssp = DSSP(model, pdb_path, dssp='mkdssp')
        
        # Map DSSP codes to simplified secondary structure
        ss_map = {
            'H': 'H',  # Alpha helix
            'B': 'E',  # Beta bridge
            'E': 'E',  # Extended strand
            'G': 'H',  # 3-10 helix
            'I': 'H',  # Pi helix
            'T': 'L',  # Turn
            'S': 'L',  # Bend
            '-': 'L',  # Loop/irregular
        }
        
        # Extract secondary structure for each residue
        for chain in model:
            chain_id = chain.id
            for residue in chain:
                if residue.id[0] == ' ':  # Only standard residues
                    res_id = str(residue.id[1])
                    key = (chain_id, (' ', residue.id[1], ' '))
                    if key in dssp:
                        ss = ss_map.get(dssp[key][2], 'L')  # Default to loop if unknown
                    else:
                        ss = 'L'  # Default to loop if not found in DSSP
                    # print(res_id, ss)
                    ss_info[chain_id][res_id] = ss
                    
    except Exception as e:
        print(f"Warning: DSSP calculation failed - {str(e)}")
        # If DSSP fails, mark all residues as loops
        for chain in structure[0]:
            chain_id = chain.id
            for residue in chain:
                if residue.id[0] == ' ':
                    res_id = str(residue.id[1])
                    ss_info[chain_id][res_id] = 'L'
    
    return ss_info

def filter_fragment_by_ss(fragment, ss_info, chain_id, args):
    """Filter fragment based on secondary structure rules."""
    # Get secondary structure sequence for the fragment
    ss_seq = [ss_info[chain_id][str(res_id)] for res_id in fragment]
    # print(fragment)
    # print(ss_seq)
    
    # Trim prefix loops except one
    start_idx = 0
    for i, ss in enumerate(ss_seq):
        if ss != 'L':
            if i > 0:  # Keep one loop residue if there are prefix loops
                start_idx = i - 1
            break
    
    # Trim suffix loops except one
    end_idx = len(ss_seq)
    for i in range(len(ss_seq) - 1, -1, -1):
        if ss_seq[i] != 'L':
            if i < len(ss_seq) - 1:  # Keep one loop residue if there are suffix loops
                end_idx = i + 2
            break
    
    # Get trimmed fragment and its SS sequence
    trimmed_fragment = fragment[start_idx:end_idx]
    trimmed_ss_seq = ss_seq[start_idx:end_idx]
    
    # print(trimmed_fragment)
    # print(trimmed_ss_seq)
    
    # Check loop fraction
    loop_count = sum(1 for ss in trimmed_ss_seq if ss == 'L')
    loop_fraction = loop_count / len(trimmed_ss_seq) if trimmed_ss_seq else 1.0
    # print(loop_fraction)
    
    # Apply filtering rules
    if len(trimmed_fragment) < 2:
        # print('filtered')
        return None
    if loop_fraction > args.max_loop_fraction:
        # print('filtered')
        return None
    if len(trimmed_fragment) < 3 and 'H' in trimmed_ss_seq:
        # print('filtered')
        return None
        
    return trimmed_fragment


def analyze_single_environments(pdb_path, output_dir, min_plddt=70, env_distance=10.0,
                              env_plddt_threshold=70, max_fragments=4, min_residues=10, args= None):
    """Analyze environments around single residues with fragment filtering."""
    
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('temp', pdb_path)
    pdb_id = os.path.basename(pdb_path).split('.')[0]
    
    # Create subdirectory for PDB outputs if it doesn't exist
    pdb_output_dir = os.path.join(output_dir, 'pdb_environments')
    os.makedirs(pdb_output_dir, exist_ok=True)
    
    # Calculate secondary structure
    ss_info = calculate_ss_for_structure(structure, pdb_path)
    
    # Get all residues with sufficient pLDDT
    single_envs = []
    filtered_stats = defaultdict(int)
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            for residue in chain:
                if residue.id[0] != ' ':  # Skip hetero-residues
                    continue
                
                # Check pLDDT
                if get_residue_plddt(residue) < min_plddt:
                    filtered_stats['low_plddt'] += 1
                    continue
                
                try:
                    ca_atom = residue['CA']
                except KeyError:
                    filtered_stats['no_ca_atom'] += 1
                    continue
                
                # Create selector for this residue
                selector = SingleNeighborhoodSelect(residue, distance_threshold=env_distance)
                
                # Get all residues in the environment
                all_residues = set()
                env_plddt_scores = []
                
                for chain2 in model:
                    for residue2 in chain2:
                        if selector.accept_residue(residue2):
                            all_residues.add((chain2.id, residue2.id[1]))
                            env_plddt_scores.append(get_residue_plddt(residue2))
                
                # Get fragments before filtering
                fragments, _ = check_sequence_gaps([r[1] for r in all_residues])
                
                # Apply fragment filtering
                filtered_fragments = []
                for fragment in fragments:
                    filtered_fragment = filter_fragment_by_ss(fragment, ss_info, chain_id, args)
                    if filtered_fragment is not None:
                        filtered_fragments.append(filtered_fragment)
                
                # Apply environment filtering criteria
                if not filtered_fragments:
                    filtered_stats['no_valid_fragments'] += 1
                    continue
                
                total_residues = sum(len(frag) for frag in filtered_fragments)
                if total_residues < min_residues:
                    filtered_stats['too_few_residues'] += 1
                    continue
                
                if len(filtered_fragments) > max_fragments:
                    filtered_stats['too_many_fragments'] += 1
                    continue
                
                min_env_plddt = np.min(env_plddt_scores)
                if min_env_plddt < env_plddt_threshold:
                    filtered_stats['low_env_plddt'] += 1
                    continue
                
                # Store environment information
                env_info = {
                    'central_residue': {
                        'chain': chain_id,
                        'resid': residue.id[1]
                    },
                    'fragments': filtered_fragments,
                    'total_residues': total_residues,
                    'min_env_plddt': float(min_env_plddt),
                    'num_fragments': len(filtered_fragments)
                }
                
                # Save this environment as a PDB file
                pdb_filename = save_single_environment(structure, env_info, pdb_output_dir, pdb_id)
                env_info['pdb_filename'] = pdb_filename
                
                single_envs.append(env_info)
    
    return single_envs, filtered_stats

def find_far_single_envs(structure, single_envs, min_distance=10.0, max_distance=15.0):
    """Find pairs of single environments with central residues between min and max distance."""
    pairs = []
    
    # Get CA coordinates for all central residues
    central_coords = []
    for env in single_envs:
        chain = structure[0][env['central_residue']['chain']]
        residue = chain[env['central_residue']['resid']]
        try:
            ca_coord = residue['CA'].get_coord()
            central_coords.append(ca_coord)
        except KeyError:
            central_coords.append(None)
    
    # Find pairs within distance range
    for i, j in combinations(range(len(single_envs)), 2):
        if central_coords[i] is None or central_coords[j] is None:
            continue
            
        dist = np.sqrt(np.sum((central_coords[i] - central_coords[j])**2))
        if min_distance <= dist <= max_distance:
            pairs.append((single_envs[i], single_envs[j], float(dist)))
    
    return pairs

def convert_to_intervals(residue_str, chain_id, total_residues):
    """Convert a string of residue IDs to interval notation with chain IDs, including gap lengths and fragment lengths."""
    if not residue_str:
        return "", 0
    
    residues = [int(x) for x in residue_str.split('_')]
    if not residues:
        return "", 0
        
    result_parts = []
    current_interval = [residues[0]]
    
    # Add length before first fragment (distance from sequence start) only if non-zero
    prefix_len = residues[0] - 1
    if prefix_len > 0:
        result_parts.append(str(prefix_len))
    
    for i in range(1, len(residues)):
        if residues[i] == residues[i-1] + 1:
            current_interval.append(residues[i])
        else:
            # Add current interval with its length
            if len(current_interval) > 1:
                result_parts.append(f"{chain_id}{current_interval[0]}-{current_interval[-1]}")
            else:
                result_parts.append(f"{chain_id}{current_interval[0]}")
                
            # Add gap length
            gap = residues[i] - residues[i-1] - 1
            result_parts.append(str(gap))
            
            current_interval = [residues[i]]
    
    # Handle the last interval with its length
    if len(current_interval) > 1:
        result_parts.append(f"{chain_id}{current_interval[0]}-{current_interval[-1]}")
    else:
        result_parts.append(f"{chain_id}{current_interval[0]}")
    
    # Add length after last fragment (distance to sequence end) only if non-zero
    suffix_len = total_residues - residues[-1]
    if suffix_len > 0:
        result_parts.append(str(suffix_len))
    
    # Calculate total sequence span (including prefix and suffix)
    total_span = residues[-1] - residues[0] + 1 + prefix_len + suffix_len
    
    return '/'.join(result_parts), total_span

def create_united_intervals(env1_resid_str, env2_resid_str, chain_id, total_residues):
    """Create united intervals from two environments, properly handling the gap between them."""
    if not env1_resid_str or not env2_resid_str:
        return ""
    
    # Convert residue strings to sorted lists of integers
    env1_residues = sorted([int(x) for x in env1_resid_str.split('_')])
    env2_residues = sorted([int(x) for x in env2_resid_str.split('_')])
    
    # Combine and sort all residues
    all_residues = sorted(set(env1_residues + env2_residues))
    
    result_parts = []
    current_interval = [all_residues[0]]
    
    # Add prefix length only if non-zero
    prefix_len = all_residues[0] - 1
    if prefix_len > 0:
        result_parts.append(str(prefix_len))
    
    for i in range(1, len(all_residues)):
        if all_residues[i] == all_residues[i-1] + 1:
            current_interval.append(all_residues[i])
        else:
            # Add current interval
            if len(current_interval) > 1:
                result_parts.append(f"{chain_id}{current_interval[0]}-{current_interval[-1]}")
            else:
                result_parts.append(f"{chain_id}{current_interval[0]}")
            
            # Add gap length
            gap = all_residues[i] - all_residues[i-1] - 1
            result_parts.append(str(gap))
            
            current_interval = [all_residues[i]]
    
    # Handle the last interval
    if len(current_interval) > 1:
        result_parts.append(f"{chain_id}{current_interval[0]}-{current_interval[-1]}")
    else:
        result_parts.append(f"{chain_id}{current_interval[0]}")
    
    # Add suffix length only if non-zero
    suffix_len = total_residues - all_residues[-1]
    if suffix_len > 0:
        result_parts.append(str(suffix_len))
    
    return '/'.join(result_parts)

def count_total_residues(structure):
    """Count total number of standard residues in the structure."""
    total = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':  # Only count standard residues
                    total += 1
    return total

def count_env_stats(residue_str):
    """Count number of fragments and total residues in an environment."""
    if not residue_str:
        return 0, 0
    
    residues = [int(x) for x in residue_str.split('_')]
    if not residues:
        return 0, 0
    
    # Count fragments by looking for gaps in sequence
    fragments = 1
    for i in range(1, len(residues)):
        if residues[i] != residues[i-1] + 1:
            fragments += 1
            
    return fragments, len(residues)

def main():
    parser = argparse.ArgumentParser(description='Analyze environments around single residues and find close pairs')
    parser.add_argument('pdb_dir', type=str, help='Directory containing PDB files')
    parser.add_argument('--pdbs_list', type=str, help='List of PDB files to get from path, if not provided, all pdbs in pdb_dir will be used', default=None)
    parser.add_argument('--output_dir', type=str, default='single_and_duo_environments',
                        help='Directory to save output files')
    
    # Single environment parameters
    parser.add_argument('--env_distance', type=float, default=10.0,
                        help='Distance threshold for environment selection')
    parser.add_argument('--min_plddt', type=float, default=70.0,
                        help='Minimum pLDDT score for individual residues')
    parser.add_argument('--env_plddt_threshold', type=float, default=70.0,
                        help='Minimum pLDDT score for the environment')
    parser.add_argument('--max_fragments', type=int, default=4,
                        help='Maximum number of fragments allowed in environment')
    parser.add_argument('--min_residues', type=int, default=10,
                        help='Minimum number of residues in environment')
    parser.add_argument('--max_loop_fraction', type=float, default=0.5,
                        help='Maximum fraction of loop residues in fragment')
    # Pair finding parameters
    parser.add_argument('--min_pair_distance', type=float, default=30.0,
                        help='Minimum distance between pairs of environments')
    parser.add_argument('--max_pair_distance', type=float, default=2000.0,
                        help='Maximum distance between pairs of environments')
    
    parser.add_argument('--stats_output', type=str, default='stats_single_and_duo_envs.json',
                        help='Output file for filtering statistics')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all PDB files
    if args.pdbs_list:
        with open(args.pdbs_list, 'r') as f:
            pdb_file_names = [line.strip() for line in f.readlines()]
        pdb_files = [f for f in os.listdir(args.pdb_dir) if f.endswith('.pdb') and f in pdb_file_names]
        print(f"Processing {len(pdb_files)} PDB files from list")
    else:
        pdb_files = [f for f in os.listdir(args.pdb_dir) if f.endswith('.pdb')]
        print(f"Processing {len(pdb_files)} PDB files from directory")
    
    # Process each PDB file
    all_results = {}
    total_filtered_stats = defaultdict(int)
    
    # Create list to store pair data
    pair_data = []
    
    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        pdb_path = os.path.join(args.pdb_dir, pdb_file)
        pdb_id = os.path.basename(pdb_file).split('.')[0]
        
        try:
            # Parse structure first to get total residue count
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('temp', pdb_path)
            total_residues = count_total_residues(structure)
            
            # First analyze single environments
            single_envs, filtered_stats = analyze_single_environments(
                pdb_path, args.output_dir,
                min_plddt=args.min_plddt,
                env_plddt_threshold=args.env_plddt_threshold,
                env_distance=args.env_distance,
                max_fragments=args.max_fragments,
                min_residues=args.min_residues,
                args=args
            )
            
            # Update filtering statistics
            for key, value in filtered_stats.items():
                total_filtered_stats[f"single_{key}"] += value
            
            if single_envs:
                # Find pairs of environments within distance threshold
                env_pairs = find_far_single_envs(structure, single_envs, args.min_pair_distance, args.max_pair_distance)
                
                # Store pair data
                for pair in env_pairs:
                    env1, env2, distance = pair
                    env1_resid_str = '_'.join(str(res_id) for fragment in env1['fragments'] for res_id in fragment)
                    env2_resid_str = '_'.join(str(res_id) for fragment in env2['fragments'] for res_id in fragment)
                    
                    # Get fragment and residue counts
                    env1_frag_count, env1_res_count = count_env_stats(env1_resid_str)
                    env2_frag_count, env2_res_count = count_env_stats(env2_resid_str)
                    
                    # Convert to interval notation with embedded gap lengths
                    env1_intervals, env1_span = convert_to_intervals(env1_resid_str, env1['central_residue']['chain'], total_residues)
                    env2_intervals, env2_span = convert_to_intervals(env2_resid_str, env2['central_residue']['chain'], total_residues)
                    
                    # Create united intervals
                    united_intervals = create_united_intervals(env1_resid_str, env2_resid_str, env1['central_residue']['chain'], total_residues)
                    
                    pair_data.append({
                        'pdb_id': pdb_id,
                        'total_pdb_residues': total_residues,
                        'env1_central_residue': env1['central_residue']['resid'],
                        'env2_central_residue': env2['central_residue']['resid'],
                        'env1_resid': env1_resid_str,
                        'env2_resid': env2_resid_str,
                        'env1_intervals': env1_intervals,
                        'env2_intervals': env2_intervals,
                        'env1_total_span': env1_span,
                        'env2_total_span': env2_span,
                        'env1_fragment_count': env1_frag_count,
                        'env2_fragment_count': env2_frag_count,
                        'env1_residue_count': env1_res_count,
                        'env2_residue_count': env2_res_count,
                        'distance': distance,
                        'united_intervals': united_intervals
                    })
                
                # Store results
                all_results[pdb_id] = {
                    'single_envs': single_envs,
                    'env_pairs': [
                        {
                            'env1': pair[0],
                            'env2': pair[1],
                            'distance': pair[2]
                        }
                        for pair in env_pairs
                    ]
                }
                
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
            continue
    
    # Save results
    output_file = os.path.join(args.stats_output)
    with open(output_file, 'w') as f:
        json.dump({
            'results': all_results,
            'filtering_stats': dict(total_filtered_stats)
        }, f, indent=2)
    
    # Save pair data to CSV
    pair_df = pd.DataFrame(pair_data)
    pair_csv = os.path.join(args.output_dir, 'paired_environments.csv')
    pair_df.to_csv(pair_csv, index=False)
    
    print(f"\nAnalysis results saved to {output_file}")
    print(f"Paired environments data saved to {pair_csv}")
    
    # Calculate and print summary statistics
    total_single_envs = sum(len(result['single_envs']) for result in all_results.values())
    total_pairs = sum(len(result['env_pairs']) for result in all_results.values())
    
    print("\nSummary Statistics:")
    print(f"Total protein structures analyzed: {len(pdb_files)}")
    print(f"Total single environments found: {total_single_envs}")
    print(f"Total environment pairs found: {total_pairs}")
    
    print("\nFiltering Statistics:")
    for reason, count in total_filtered_stats.items():
        print(f"Environments filtered due to {reason}: {count}")

if __name__ == '__main__':
    main() 
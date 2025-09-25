import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
import os
import argparse
import pandas as pd
from tqdm import tqdm
from utils.metrics_rmsd import calculate_rmsd, calculate_scrmsd
from utils.metrics_plddt import get_plddt
from utils.result_scores import get_sr


def main(args):

    if args.model_type == 'struct':
        assert args.self_cons_dir is not None, "self-cons-dir is required for structure models"
        assert args.num_sc_seqs is not None, "num-sc-seqs is required for structure models"

    # Read metadata CSV
    df = pd.read_csv(args.input_csv)
    
    res_df = df.copy()
    print(f"Found {len(df.entry.unique())} entries with {len(df.iteration.unique())} iterations")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        
        iteration = row['iteration']
        # pdb_id = row['pdb_id']
        pdb_id = row['entry'].split('_')[1]

        chain_id = 'A'
        gen_motif_residues = [int(x) for x in row['gen_res_ids'].split('_')]
        ref_motif_residues = [int(x) for x in row['ref_res_ids'].split('_')]

        g_name = sc_name = f'{row["gen_pdb_name"]}'
        ref_pdb = os.path.join(args.ref_pdb_dir, pdb_id + '.pdb')
        gen_pdb = os.path.join(args.gen_pdb_dir, f"{g_name}.pdb")


        if args.model_type == 'struct':
            sc_pdb_base = os.path.join(args.self_cons_dir, sc_name)

        # Check if reference and generated files exist
        if not os.path.exists(ref_pdb):
            print(f"Skipping because of missing reference: {ref_pdb}")
            continue
        if not os.path.exists(gen_pdb):
            print(f"Skipping because of missing generated: {gen_pdb}")
            continue

        if args.model_type == 'struct':
            # calculate scRMSD
            best_sc_rmsd, best_sc_pdb = calculate_scrmsd(
                ref_pdb, gen_pdb, sc_pdb_base, gen_motif_residues, ref_motif_residues, args.num_sc_seqs
            )
            res_df.at[idx, 'scrmsd'] = best_sc_rmsd  # index of best of num_sc_seqs non-motif RMSD between generated and self-consistency
            best_seq = int(os.path.basename(best_sc_pdb).split('_seq')[-1].split('.')[0])
            res_df.at[idx, 'best_seq'] = best_seq

            # calculate motif RMSD
            motif_rmsd = calculate_rmsd(
                ref_pdb, best_sc_pdb, ref_motif_residues, gen_motif_residues
            )
        else:
            # calculate plddt
            plddt = get_plddt(gen_pdb)
            res_df.at[idx, 'plddt'] = plddt

            # calculate motif RMSD
            motif_rmsd = calculate_rmsd(
                ref_pdb, gen_pdb, ref_motif_residues, gen_motif_residues
            )

        if motif_rmsd is not None:
            res_df.at[idx, 'rmsd'] = motif_rmsd  # motif RMSD between ref and best self-consistency

    # Save results
    res_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate RMSDs for sequence and structure models')
    parser.add_argument('--input-csv', required=True, help='Input CSV file with model metadata, metadata should include [entry, gen_pdb_name, iteration, gen_res_ids, ref_res_ids, total_len]')
    parser.add_argument('--output-csv', required=True, help='Output CSV file for results')

    parser.add_argument('--ref-pdb-dir', required=True, help='Directory containing reference PDB files')
    parser.add_argument('--gen-pdb-dir', required=True, help='Directory containing generated PDB files')

    parser.add_argument('--model-type', required=True, help='Type of model, either "seq" or "struct"')

    # struct model specific
    parser.add_argument('--self-cons-dir', required=False, help='Directory containing self-consistency PDB files')
    parser.add_argument('--num-sc-seqs', required=False , type=int, default=8, help='Number of self-consistency sequences for structure models')
    args = parser.parse_args()
    main(args)
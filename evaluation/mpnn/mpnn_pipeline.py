import argparse
import os
import glob
import subprocess
import json
import math
import copy
from multiprocessing import Pool
import torch
import torch.multiprocessing as mp
from ProteinMPNN.protein_mpnn_run import main as single_gpu_main
import sys
from pathlib import Path
from tqdm import tqdm
import shutil
def remove_tmp(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)

def parse_pdb(args_tuple):
    pdb_path, output_dir, work_dir, ca_only = args_tuple
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
    output_path = os.path.join(output_dir, f"{pdb_id}.jsonl")
    cmd = [
        "python", os.path.join(work_dir, "parse_multiple_chains.py"),
        f"--input_file={pdb_path}",
        f"--output_path={output_dir}"
    ]
    if ca_only:
        cmd.append("--ca_only")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {pdb_path}: {e}")
    return output_path

def parse_all_pdbs(exp_name, pdbs_folder, work_dir, out_folder, n_procs, ca_only):
    os.makedirs(out_folder, exist_ok=True)
    tmp_dir = os.path.join(out_folder, f"{exp_name}_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    pdb_files = glob.glob(os.path.join(pdbs_folder, "**", "*.pdb"), recursive=True)
    pdb_files = sorted(pdb_files)
    print(f"Found {len(pdb_files)} pdb files.")
    pool_args = [(pdb_path, tmp_dir, work_dir, ca_only) for pdb_path in pdb_files]
    with Pool(processes=n_procs) as pool:
        pool.map(parse_pdb, pool_args)
    out_jsonl = os.path.join(out_folder, f"{exp_name}.jsonl")
    with open(out_jsonl, "w") as outfile:
        for pdb_path in pdb_files:
            pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
            jsonl_path = os.path.join(tmp_dir, f"{pdb_id}.jsonl")
            if os.path.exists(jsonl_path):
                with open(jsonl_path, "r") as infile:
                    for line in infile:
                        outfile.write(line)
    print(f"Parsed {exp_name}")
    return out_jsonl

def split_dataset(jsonl_path, total_chunks):
    with open(jsonl_path, 'r') as f:
        data = [json.loads(line) for line in f]
    print('found', len(data), 'pdbs')
    chunk_size = math.ceil(len(data) / total_chunks)
    print('chunk_size', chunk_size)
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunk_path = f"{jsonl_path}.chunk_{len(chunks)}"
        with open(chunk_path, 'w') as f:
            for item in chunk:
                json.dump(item, f)
                f.write('\n')
        chunks.append(chunk_path)
    return chunks

def mpnn_worker(gpu_id, process_id, args, chunk_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    worker_args = copy.deepcopy(args)
    worker_args.jsonl_path = chunk_path
    worker_args.out_folder = os.path.join(args.tmp_folder, f'gpu_{gpu_id}_process_{process_id}')
    single_gpu_main(worker_args)

def run_mpnn_parallel(args):
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        print("No GPUs available. Running on CPU.")
        single_gpu_main(args)
        return
    processes_per_gpu = args.processes_per_gpu
    total_processes = num_gpus * processes_per_gpu
    print(f"Running on {num_gpus} GPUs with {processes_per_gpu} processes per GPU (total {total_processes} processes)")
    mp.set_start_method('spawn', force=True)
    chunk_paths = split_dataset(args.jsonl_path, total_processes)
    processes = []
    for gpu_id in range(num_gpus):
        for proc_id in range(processes_per_gpu):
            chunk_idx = gpu_id * processes_per_gpu + proc_id
            if chunk_idx < len(chunk_paths):
                p = mp.Process(
                    target=mpnn_worker,
                    args=(gpu_id, proc_id, args, chunk_paths[chunk_idx])
                )
                p.start()
                processes.append(p)
    for p in processes:
        p.join()
    for chunk_path in chunk_paths:
        try:
            os.remove(chunk_path)
        except:
            pass
    print("All MPNN processes completed")

def process_fasta(fasta_path):
    sequences = []
    current_seq = {'header': '', 'sequence': ''}
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq['header']:
                    sequences.append(current_seq)
                current_seq = {'header': line[1:], 'sequence': ''}
            else:
                current_seq['sequence'] += line
    if current_seq['header']:
        sequences.append(current_seq)
    return sequences[1:] if len(sequences) > 1 else []

def postprocess_sequences(out_folder, exp_name):
    output_file_path = os.path.join(out_folder, f"{exp_name}_mpnn8.fasta")
    seqs_dirs = []
    for root, dirs, files in os.walk(out_folder):
        for d in dirs:
            if d == 'seqs':
                seqs_dirs.append(os.path.join(root, d))
    with open(output_file_path, 'w') as output_file:
        for seqs_dir in seqs_dirs:
            for file in os.listdir(seqs_dir):
                if file.endswith('.fa'):
                    fasta_path = os.path.join(seqs_dir, file)
                    basename = Path(file).stem
                    remaining_seqs = process_fasta(fasta_path)
                    if remaining_seqs:
                        for idx, seq in enumerate(remaining_seqs, start=1):
                            output_file.write(f">{basename}_seq{idx}\n{seq['sequence']}\n")
    print(f"Postprocessed sequences written to {output_file_path}")

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Parsing args
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--pdbs_folder", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--n_procs", type=int, default=4)
    parser.add_argument("--ca_only", action="store_true", default=False)
    parser.add_argument("--skip_parse", action="store_true", default=False, help="Skip PDB parsing step if already done")
    # MPNN args
    parser.add_argument("--processes_per_gpu", type=int, default=1)
    parser.add_argument("--suppress_print", type=int, default=0)
    parser.add_argument("--path_to_model_weights", type=str, default="")
    parser.add_argument("--model_name", type=str, default="v_48_020")
    parser.add_argument("--use_soluble_model", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_score", type=int, default=0)
    parser.add_argument("--save_probs", type=int, default=0)
    parser.add_argument("--score_only", type=int, default=0)
    parser.add_argument("--path_to_fasta", type=str, default="")
    parser.add_argument("--conditional_probs_only", type=int, default=0)
    parser.add_argument("--conditional_probs_only_backbone", type=int, default=0)
    parser.add_argument("--unconditional_probs_only", type=int, default=0)
    parser.add_argument("--backbone_noise", type=float, default=0.00)
    parser.add_argument("--num_seq_per_target", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=200000)
    parser.add_argument("--sampling_temp", type=str, default="0.1")
    parser.add_argument("--pdb_path", type=str, default='')
    parser.add_argument("--pdb_path_chains", type=str, default='')
    parser.add_argument("--chain_id_jsonl",type=str, default='')
    parser.add_argument("--fixed_positions_jsonl", type=str, default='')
    parser.add_argument("--omit_AAs", type=list, default='X')
    parser.add_argument("--bias_AA_jsonl", type=str, default='')
    parser.add_argument("--bias_by_res_jsonl", default='')
    parser.add_argument("--omit_AA_jsonl", type=str, default='')
    parser.add_argument("--pssm_jsonl", type=str, default='')
    parser.add_argument("--pssm_multi", type=float, default=0.0)
    parser.add_argument("--pssm_threshold", type=float, default=0.0)
    parser.add_argument("--pssm_log_odds_flag", type=int, default=0)
    parser.add_argument("--pssm_bias_flag", type=int, default=0)
    parser.add_argument("--tied_positions_jsonl", type=str, default='')
    args = parser.parse_args()

    args.tmp_folder = os.path.join(args.out_folder, "tmp")
    # Stage 1: Parse PDBs
    if not args.skip_parse:
        jsonl_path = parse_all_pdbs(args.exp_name, args.pdbs_folder, args.work_dir, args.tmp_folder, args.n_procs, args.ca_only)
    else:
        jsonl_path = os.path.join(args.tmp_folder, f"{args.exp_name}.jsonl")
    # Stage 2: Run MPNN
    args.jsonl_path = jsonl_path
    run_mpnn_parallel(args)
    # Stage 3: Postprocess sequences
    postprocess_sequences(args.out_folder, args.exp_name)

    remove_tmp(args.tmp_folder)

if __name__ == "__main__":
    main() 
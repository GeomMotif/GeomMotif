import argparse
import os
import glob
import subprocess
from multiprocessing import Pool

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("pdbs_folder", type=str)
    parser.add_argument("work_dir", type=str)
    parser.add_argument("out_folder", type=str)
    parser.add_argument("n_procs", type=int)
    parser.add_argument("--ca_only", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)
    tmp_dir = os.path.join(args.out_folder, f"{args.exp_name}_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    pdb_files = glob.glob(os.path.join(args.pdbs_folder, "**", "*.pdb"), recursive=True)
    pdb_files = sorted(pdb_files)
    print(f"Found {len(pdb_files)} pdb files.")

    # Prepare arguments for each process
    pool_args = [(pdb_path, tmp_dir, args.work_dir, args.ca_only) for pdb_path in pdb_files]

    with Pool(processes=args.n_procs) as pool:
        pool.map(parse_pdb, pool_args)

    # Concatenate all jsonl files
    out_jsonl = os.path.join(args.out_folder, f"{args.exp_name}.jsonl")
    with open(out_jsonl, "w") as outfile:
        for pdb_path in pdb_files:
            pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
            jsonl_path = os.path.join(tmp_dir, f"{pdb_id}.jsonl")
            if os.path.exists(jsonl_path):
                with open(jsonl_path, "r") as infile:
                    for line in infile:
                        outfile.write(line)

    print(f"Parsed {args.exp_name}")

if __name__ == "__main__":
    main() 
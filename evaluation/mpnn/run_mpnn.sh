#!/bin/bash

# exp=$1
# pdbs_folder=$2
# out_folder=$3

# # 8 sequences per target
# n_seq=$4
# # processes for parsing chains
# n_procs=$5
# # Processes per each GPU, usually each mpnn utilizes 0.25-0.3 of GPU. Code uses all available GPUs.
# processes_per_gpu=$6


# exp="test"

python mpnn_pipeline.py \
    --exp_name $exp \
    --num_seq_per_target 8 \
    --n_procs 10 \
    --processes_per_gpu 4 \
    --pdbs_folder ../example/struct/gen_pdbs \
    --out_folder ../example/struct/sc_pdbs_ \
    --work_dir $(realpath ./)
    
    
    # --ca_only
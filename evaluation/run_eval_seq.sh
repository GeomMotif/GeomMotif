exp=test_seq
gen_pdbs_folder=$(realpath example/seq/gen_pdbs)
meta_csv=$(realpath example/seq/test_generated.csv)
res_csv=$(realpath example/seq/test_res_table.csv)

set -e  # Exit on any error

### Sequence generation models evaluation: 
# 1- run ESMFold:
torchrun  \
        --nproc_per_node=8 \
        --master_port=34568 \
        -m folding.structure_from_sequence \
        --pdb_path="${gen_pdbs_folder}" \
        --input-file="${meta_csv}" \
        --name-col gen_pdb_name \
        --batch_size=4

# 2- get plddt and calculate RMSD:
python metrics_calculation.py \
        --model-type seq \
        --input-csv "${meta_csv}sxs" \
        --output-csv "${res_csv}" \
        --ref-pdb-dir ref_pdbs \
        --gen-pdb-dir "${gen_pdbs_folder}"

# 3- calculate novelty and diversity:
python novelty_and_diversity.py \
    --input-csv "${res_csv}" \
    --output-csv "${res_csv}" \
    --gen-pdb-dir "${gen_pdbs_folder}" \
    --ref-pdb-dir ref_pdbs \
    --n_cors 8

# 4- show final scores:
python print_sun_score.py \
    --input-csv "${res_csv}"
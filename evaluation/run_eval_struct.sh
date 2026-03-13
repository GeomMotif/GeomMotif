### Structure generation models evaluation: 

exp=test_struct
gen_pdbs_folder=$(realpath example/struct/gen_pdbs)
sc_pdbs_folder=$(realpath example/struct/sc_pdbs_)
meta_csv=$(realpath example/struct/test_meta.csv)
res_csv=$(realpath example/struct/test_res_table_mpnn8.csv)

set -e  # Exit on any error
### Structure generation models evaluation: 

# 1- run mpnn:
cd mpnn
python mpnn_pipeline.py \
    --exp_name $exp \
    --num_seq_per_target 8 \
    --n_procs 10 \
    --processes_per_gpu 4 \
    --pdbs_folder $gen_pdbs_folder \
    --out_folder $sc_pdbs_folder \
    --work_dir $(realpath ./)

cd ..

# 2- run ESMFold on the mpnn generated:
torchrun  \
        --nproc_per_node=8 \
        --master_port=34568 \
        -m folding.structure_from_sequence \
        --pdb_path="${sc_pdbs_folder}/" \
        --input-file="${sc_pdbs_folder}/${exp}_mpnn8.fasta" \
        --batch_size=4


# 3- calculate RMSD and scRMSD:
python metrics_calculation.py \
        --model-type struct \
        --input-csv ${meta_csv} \
        --output-csv ${res_csv} \
        --ref-pdb-dir ref_pdbs \
        --gen-pdb-dir ${gen_pdbs_folder} \
        --self-cons-dir ${sc_pdbs_folder} \
        --num-sc-seqs 8

# 4- calculate novelty and diversity:
python novelty_and_diversity.py \
    --input-csv ${res_csv} \
    --output-csv ${res_csv} \
    --gen-pdb-dir ${gen_pdbs_folder} \
    --ref-pdb-dir ref_pdbs \
    --n_cors 8

# 5- show final scores:
python print_sun_score.py \
    --input-csv ${res_csv} \
    --meta-csv data/GeomMotif/task_conditions.csv

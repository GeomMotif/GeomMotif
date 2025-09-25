
output_dir=./processed_res
pdb_dir=./test_ref_pdbs
# rm -rf $output_dir
mkdir -p $output_dir
# 1
# python get_pdbs_by_filtered_list.py

# 2 clustering
python run_pairwise_TM.py \
    $pdb_dir \
    $output_dir/stage_2_pairwise_TM_cov03.csv \
    --n_workers 10 \
    --min_coverage 0.3

python cluster.py \
    $output_dir/stage_2_pairwise_TM_cov03_matrix.npz \
    --output_prefix $output_dir/stage_2_cov03 \
    --methods complete

python sample_protein_from_cluster.py \
    $output_dir/stage_2_cov03_complete_assignments.txt \
    $output_dir/stage_2_cov03_sampled_proteins.txt \
    --tm_cutoff 0.5

# 3 4 5: get single and paired motifs and filter them
python find_substructures.py \
    $pdb_dir \
    --pdbs_list $output_dir/stage_2_cov03_sampled_proteins.txt \
    --output_dir $output_dir/motifs \
    --env_distance 13 \
    --min_pair_distance 30 \
    --max_pair_distance 200 \
    --min_plddt 0 \
    --env_plddt_threshold 0 \
    --max_fragments 100 \
    --min_residues 30 \
    --max_loop_fraction 0.25 \
    --stats_output $output_dir/motifs_stats.json

# 6 sample motifs by number of fragments
python sample_motifs.py
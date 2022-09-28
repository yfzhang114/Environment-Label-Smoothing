
dataset='HHAR'
# CUDA_VISIBLE_DEVICES=3  nohup python3 -m woods.scripts.hparams_sweep \
#         --dataset $dataset\
#         --objective DANN \
#         --n_hparams 100\
#         --save_path ./results_dann_$dataset \
#         --launcher local  > sweep_DANN_$dataset.out 2>&1 &
        
# python3 -m woods.scripts.compile_results \
#         --mode tables \
#         --results_dir ./results_dann_LSA64_ls \
#         --latex

python3 -m woods.scripts.compile_results \
        --results_dir logs/hhar/results_dann_HHAR_ls_hparams_20 \
        --mode hparams
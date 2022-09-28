# dataset='amazon'
# alg='DANN'
# seed=0
# eps=0.3
# CUDA_VISIBLE_DEVICES=2 nohup python examples/run_expt.py --dataset $dataset --label_smooth True --eps $eps --seed $seed --algorithm $alg --root_dir Data_Path >$alg-$dataset-seed$seed-ls$eps.out 2>&1 & 
# ogb-molpcba
# for((i=0;i<=0;i++));  
# do
# test_env=2
# seed=1
# CUDA_VISIBLE_DEVICES=3 nohup python train.py  --dataset $dataset --algorithm $algorithm --label_smooth --eps 0.6 --test_envs $test_env --seed $seed --output_dir logs/$algorithm$dataset$seed > $algorithm$dataset$test_env$seed-ls06.out 2>&1 &
# done
# amazon
dataset='ogb-molpcba'
alg='DANN'
seed=0
eps=0.3
CUDA_VISIBLE_DEVICES=2 python examples/run_expt.py --log_dir ./ogb --dataset ogb-molpcba --algorithm DANN --seed 0 --unlabeled_split test_unlabeled --groupby_fields from_source_domain --gradient_accumulation_steps 4 --dann_classifier_lr 0.008171478915115861 --dann_featurizer_lr 0.0008171478915115861 --dann_discriminator_lr 0.013704648392849014 --dann_penalty_weight 0.10903843281968521 --unlabeled_batch_size 3584 --batch_size 512 --n_epochs 50 --loader_kwargs num_workers=4 pin_memory=True --unlabeled_loader_kwargs num_workers=8 pin_memory=True > ogb-dann-baseline.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python examples/run_expt.py --log_dir ./amazon --dataset amazon --algorithm DANN --seed 0 --unlabeled_split test_unlabeled --groupby_fields from_source_domain --gradient_accumulation_steps 4 --dann_classifier_lr 9.805774698139038e-05 --dann_featurizer_lr 9.805774698139038e-06 --dann_discriminator_lr 0.0001644557807141881 --dann_penalty_weight 0.1090384328nvid1968521 --unlabeled_batch_size 21 --batch_size 3 --n_epochs 1 > amazon-dann-baseline.out 2>&1 &

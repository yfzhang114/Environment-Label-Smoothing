
algorithm='DANN'
dataset='PACS'
test_env=0
seed=0
epses=0.5
for((i=0;i<=0;i++));  
do
test_env=$i
CUDA_VISIBLE_DEVICES=1 nohup python train.py --dataset $dataset  --algorithm $algorithm --test_envs $test_env --seed $seed --output_dir logs/$algorithm$dataset$test_env$seed --exp_name $algorithm$dataset$test_env$seed --label_smooth --eps $epses > $algorithm$dataset$test_env$seed-dstep5-eps$epses.out 2>&1 &#RotatedMNIST GroupDRO
done
# for((i=0;i<=2;i++));  
# do
# eps=0.3
# semi_fraction=0.01
# step_disc=500
# test_env=$i
# CUDA_VISIBLE_DEVICES=1 nohup python train_semi.py --random_label --dataset $dataset  --algorithm $algorithm --test_envs $test_env --seed $seed --output_dir  logs/semi/$algorithm$dataset$test_env$seed --semi_fraction $semi_fraction --step_disc $step_disc > $algorithm$dataset$test_env$seed-step$step_disc-semifrac$semi_fraction-BASE.out 2>&1 &
# # CUDA_VISIBLE_DEVICES=3 nohup python train.py --dataset $dataset --label_smooth --eps $eps --algorithm $algorithm --test_envs $test_env --seed $seed --output_dir logs/$algorithm$dataset$test_env$seed > $algorithm$dataset$test_env$seed-eps$eps.out 2>&1 &
# done

# # # CUDA_VISIBLE_DEVICES=2 nohup python train.py --exp_name > nohup_gradients.out 2>&1 &
# # # eps=0.7
# # # CUDA_VISIBLE_DEVICES=2 nohup python train.py  --checkpoint_freq 10 --exp_name dann-step5-eps$eps --label_smooth --eps $eps > dann-step5-eps$eps.out 2>&1 &

# # algorithm='ERMClass'
# # dataset='RotatedMNIST'
# # #RotatedMNIST
# # test_env=0
# # seed=0
# # alpha_cover=0.001
# # cover_lambda=1.0
# # alpha_n='times_domain'
# # for((i=0;i<=0;i++));  
# # do
# # seed=$i
# # CUDA_VISIBLE_DEVICES=1 nohup python train.py --dataset $dataset --algorithm $algorithm --cover --alpha_cover $alpha_cover  --cover_lambda $cover_lambda --test_envs $test_env --seed $seed --exp_name 5wlr5e-5alpha$alpha_cover-lambda$cover_lambda$seed$dataset$algorithm> 5wlr5e-5$algorithm$dataset$test_env$seed-alpha$alpha_cover-lambda$cover_lambda.out 2>&1 &
# # # CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset $dataset  --algorithm $algorithm --test_envs $test_env --seed $seed> $algorithm$dataset$test_env$seed.out 2>&1 &
# # # 5e-2: cuda 1, 5e-3: cuda 2, 5e-4: cuda 3 
# # done

# algorithm='ERM'
# dataset='RotatedMNIST'
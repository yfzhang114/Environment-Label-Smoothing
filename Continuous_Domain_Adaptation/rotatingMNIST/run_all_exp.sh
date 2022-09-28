
model=DANN
CUDA_VISIBLE_DEVICES=1 nohup python model.py --model $model > $model-eps03.out 2>&1 &
# python model.py --model PCIDA
# python model.py --model SO
# python model.py --model ADDA
# python model.py --model DANN
# python model.py --model CUA

name='a2d'
nohup python cdan.py data/office31 -d Office31 -s amazon -t dslr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --log logs/cdan_resnet50/Office31_$name --log_name $name-cdan_resnet50 --gpu 2 --lr 0.01 --bottleneck-dim 256 --eps 1. > $name-cdann-epses.out 2>&1 &

name='a2w'
nohup python cdan.py data/office31 -d Office31 -s amazon -t webcam -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --log logs/cdan_resnet50/Office31_$name --log_name $name-cdan_resnet50 --gpu 2  --lr 0.01   --bottleneck-dim 256 --eps 1. > $name-cdann-epses.out 2>&1 &

name='d2w'
nohup python cdan.py data/office31 -d Office31 -s dslr -t webcam -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --log logs/cdan_resnet50/Office31_$name --log_name $name-cdan_resnet50 --gpu 2  --lr 0.01   --bottleneck-dim 256 --eps 1. > $name-cdann-epses.out 2>&1 &

name='d2a'
nohup python cdan.py data/office31 -d Office31 -s dslr -t amazon -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --log logs/cdan_resnet50/Office31_$name --log_name $name-cdan_resnet50 --gpu 3   --lr 0.01   --bottleneck-dim 256 --eps 1. > $name-cdann-epses.out 2>&1 &

name='w2d'
nohup python cdan.py data/office31 -d Office31 -s webcam -t dslr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --log logs/cdan_resnet50/Office31_$name --log_name $name-cdan_resnet50 --gpu 3   --lr 0.01   --bottleneck-dim 256 --eps 1. > $name-cdann-epses.out 2>&1 &

name='w2a'
nohup python cdan.py data/office31 -d Office31 -s webcam -t amazon -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --log logs/cdan_resnet50/Office31_$name --log_name $name-cdan_resnet50 --gpu 3   --lr 0.01   --bottleneck-dim 256 --eps 1. > $name-cdann-epses.out 2>&1 &
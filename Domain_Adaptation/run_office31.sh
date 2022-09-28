
name='a2d'
nohup python cdan_mcc_sdat.py data/office31 -d Office31 -s amazon -t dslr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --log logs/cdan_mcc_sdat_office31_vit/Office31_$name --log_name $name-cdan_mcc_sdat_office31_vit --gpu 0 --rho 0.02 --lr 0.002 --no-pool  --temperature 2.0 --bottleneck-dim 256 --eps 1. > office31-vit-$name.out 2>&1 &

name='a2w'
nohup python cdan_mcc_sdat.py data/office31 -d Office31 -s amazon -t webcam -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --log logs/cdan_mcc_sdat_office31_vit/Office31_$name --log_name $name-cdan_mcc_sdat_office31_vit --gpu 1 --rho 0.02 --lr 0.002 --no-pool  --temperature 2.0 --bottleneck-dim 256 --eps 1. > office31-vit-$name.out 2>&1 &

name='d2w'
nohup python cdan_mcc_sdat.py data/office31 -d Office31 -s dslr -t webcam -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --log logs/cdan_mcc_sdat_office31_vit/Office31_$name --log_name $name-cdan_mcc_sdat_office31_vit --gpu 2 --rho 0.02 --lr 0.002 --no-pool  --temperature 2.0 --bottleneck-dim 256 --eps 1. > office31-vit-$name.out 2>&1 &

name='d2a'
nohup python cdan_mcc_sdat.py data/office31 -d Office31 -s dslr -t amazon -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --log logs/cdan_mcc_sdat_office31_vit/Office31_$name --log_name $name-cdan_mcc_sdat_office31_vit --gpu 3 --rho 0.02 --lr 0.002 --no-pool  --temperature 2.0 --bottleneck-dim 256 --eps 1. > office31-vit-$name.out 2>&1 &

# name='w2d'
# nohup python cdan_mcc_sdat.py data/office31 -d Office31 -s webcam -t dslr -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --log logs/cdan_mcc_sdat_office31_vit/Office31_$name --log_name $name-cdan_mcc_sdat_office31_vit --gpu 1 --rho 0.02 --lr 0.002 --no-pool  --temperature 2.0 --bottleneck-dim 256 --eps 1. > office31-vit-$name.out 2>&1 &

# name='w2a'
# nohup python cdan_mcc_sdat.py /data1/User/datasets/office31/ -d Office31 -s webcam -t amazon -a vit_base_patch16_224 --epochs 30 --seed 0 -b 24 --log logs/cdan_mcc_sdat_office31_vit/Office31_$name --log_name $name-cdan_mcc_sdat_office31_vit --gpu 3 --rho 0.02 --lr 0.002 --no-pool  --temperature 2.0 --bottleneck-dim 256 --eps 1. > office31-vit-$name.out 2>&1 &

# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 30 --seed 0 -b 24 --log logs/cdan_mcc_sdat_office31_vit/Office31_Ar2Rw --log_name Ar2Rw_cdan_mcc_sdat_office31_vit --gpu 0 --rho 0.02 --lr 0.002 --no-pool --log_results  > nohup.out 2>&1 &

# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 30 --seed 0 -b 24 --log logs/cdan_mcc_sdat_office31_vit/Office31_Cl2Ar --log_name Cl2Ar_cdan_mcc_sdat_office31_vit --gpu 0 --rho 0.02 --lr 0.002 --no-pool --log_results  > nohup.out 2>&1 &
# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 30 --seed 0 -b 24 --log logs/cdan_mcc_sdat_office31_vit/Office31_Cl2Pr --log_name Cl2Pr_cdan_mcc_sdat_office31_vit --gpu 0 --rho 0.02 --lr 0.002 --no-pool --log_results  > nohup.out 2>&1 &
# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 30 --seed 0 -b 24 --log logs/cdan_mcc_sdat_office31_vit/Office31_Cl2Rw --log_name Cl2Rw_cdan_mcc_sdat_office31_vit --gpu 0 --rho 0.02 --lr 0.002 --no-pool --log_results  > nohup.out 2>&1 &

# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 30 --seed 0 -b 24 --log logs/cdan_mcc_sdat_office31_vit/Office31_Pr2Ar --log_name Pr2Ar_cdan_mcc_sdat_office31_vit --gpu 0 --rho 0.02 --lr 0.002 --no-pool --log_results  > nohup.out 2>&1 &
# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 30 --seed 0 -b 24 --log logs/cdan_mcc_sdat_office31_vit/Office31_Pr2Cl --log_name Pr2Cl_cdan_mcc_sdat_office31_vit --gpu 0 --rho 0.02 --lr 0.002 --no-pool --log_results  > nohup.out 2>&1 &
# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 30 --seed 0 -b 24 --log logs/cdan_mcc_sdat_office31_vit/Office31_Pr2Rw --log_name Pr2Rw_cdan_mcc_sdat_office31_vit --gpu 0 --rho 0.02 --lr 0.002 --no-pool --log_results  > nohup.out 2>&1 &

# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 30 --seed 0 -b 24 --log logs/cdan_mcc_sdat_office31_vit/Office31_Rw2Ar --log_name Rw2Ar_cdan_mcc_sdat_office31_vit --gpu 0 --rho 0.02 --lr 0.002 --no-pool --log_results  > nohup.out 2>&1 &
# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 30 --seed 0 -b 24 --log logs/cdan_mcc_sdat_office31_vit/Office31_Rw2Cl --log_name Rw2Cl_cdan_mcc_sdat_office31_vit --gpu 0 --rho 0.02 --lr 0.002 --no-pool --log_results  > nohup.out 2>&1 &
# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 30 --seed 0 -b 24 --log logs/cdan_mcc_sdat_office31_vit/Office31_Rw2Pr --log_name Rw2Pr_cdan_mcc_sdat_office31_vit --gpu 0 --rho 0.02 --lr 0.002 --no-pool --log_results  > nohup.out 2>&1 &

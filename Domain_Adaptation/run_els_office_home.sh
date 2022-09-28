
# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 30 --seed 0 -b 32 --log logs/cdan_mcc_sdat_resnet50/OfficeHome_Ar2Cl_setting2 --log_name Ar2Cl_cdan_mcc_sdat_resnet50_2_seed2 --gpu 0 --rho 0.02 --lr 0.01 --temperature 2.0 --bottleneck-dim 256 --eps 1. --log_results  > a2ceps_worker2_tem2_seed2_bot256.out 2>&1 &


# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 30 --seed 0 -b 32 --log logs/cdan_mcc_sdat_resnet50/OfficeHome_Ar2Pr --log_name Ar2Pr_cdan_mcc_sdat_resnet50 --gpu 1 --rho 0.02 --lr 0.01 --log_results  --temperature 2.0 --bottleneck-dim 256 --eps 1. > a2p_eps_worker2_tem2_seed2_bot25.out 2>&1 &

# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 30 --seed 0 -b 32 --log logs/cdan_mcc_sdat_resnet50/OfficeHome_Ar2Rw --log_name A2r_cdan_mcc_sdat_resnet50 --gpu 2 --rho 0.02 --lr 0.01 --log_results  --temperature 2.0 --bottleneck-dim 256 --eps 1. > a2r_eps_worker2_tem2_seed2_bot25.out 2>&1 &

# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 30 --seed 0 -b 32 --log logs/cdan_mcc_sdat_resnet50/OfficeHome_Cl2Ar --log_name c2a_cdan_mcc_sdat_resnet50 --gpu 3 --rho 0.02 --lr 0.01 --log_results  --temperature 2.0 --bottleneck-dim 256 --eps 1. > c2a_eps_worker2_tem2_seed2_bot25.out 2>&1 &

name='c2p'
nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 30 --seed 0 -b 32 --log logs/cdan_mcc_sdat_resnet50/OfficeHome_$name --log_name $name-cdan_mcc_sdat_resnet50 --gpu 0 --rho 0.02 --lr 0.01  --temperature 2.0 --bottleneck-dim 256 --eps 1. > $name.out 2>&1 &

name='r2c'
nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 30 --seed 0 -b 32 --log logs/cdan_mcc_sdat_resnet50/OfficeHome_$name --log_name $name-cdan_mcc_sdat_resnet50 --gpu 1 --rho 0.02 --lr 0.01  --temperature 2.0 --bottleneck-dim 256 --eps 1. > $name.out 2>&1 &

# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 30 --seed 0 -b 32 --log logs/cdan_mcc_sdat_resnet50/OfficeHome_Ar2Rw --log_name Ar2Rw_cdan_mcc_sdat_resnet50 --gpu 0 --rho 0.02 --lr 0.01 --log_results  > nohup.out 2>&1 &

# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 30 --seed 0 -b 32 --log logs/cdan_mcc_sdat_resnet50/OfficeHome_Cl2Ar --log_name Cl2Ar_cdan_mcc_sdat_resnet50 --gpu 0 --rho 0.02 --lr 0.01 --log_results  > nohup.out 2>&1 &
# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 30 --seed 0 -b 32 --log logs/cdan_mcc_sdat_resnet50/OfficeHome_Cl2Pr --log_name Cl2Pr_cdan_mcc_sdat_resnet50 --gpu 0 --rho 0.02 --lr 0.01 --log_results  > nohup.out 2>&1 &
# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 30 --seed 0 -b 32 --log logs/cdan_mcc_sdat_resnet50/OfficeHome_Cl2Rw --log_name Cl2Rw_cdan_mcc_sdat_resnet50 --gpu 0 --rho 0.02 --lr 0.01 --log_results  > nohup.out 2>&1 &

# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 30 --seed 0 -b 32 --log logs/cdan_mcc_sdat_resnet50/OfficeHome_Pr2Ar --log_name Pr2Ar_cdan_mcc_sdat_resnet50 --gpu 0 --rho 0.02 --lr 0.01 --log_results  > nohup.out 2>&1 &
# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 30 --seed 0 -b 32 --log logs/cdan_mcc_sdat_resnet50/OfficeHome_Pr2Cl --log_name Pr2Cl_cdan_mcc_sdat_resnet50 --gpu 0 --rho 0.02 --lr 0.01 --log_results  > nohup.out 2>&1 &
# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 30 --seed 0 -b 32 --log logs/cdan_mcc_sdat_resnet50/OfficeHome_Pr2Rw --log_name Pr2Rw_cdan_mcc_sdat_resnet50 --gpu 0 --rho 0.02 --lr 0.01 --log_results  > nohup.out 2>&1 &

# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 30 --seed 0 -b 32 --log logs/cdan_mcc_sdat_resnet50/OfficeHome_Rw2Ar --log_name Rw2Ar_cdan_mcc_sdat_resnet50 --gpu 0 --rho 0.02 --lr 0.01 --log_results  > nohup.out 2>&1 &
# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 30 --seed 0 -b 32 --log logs/cdan_mcc_sdat_resnet50/OfficeHome_Rw2Cl --log_name Rw2Cl_cdan_mcc_sdat_resnet50 --gpu 0 --rho 0.02 --lr 0.01 --log_results  > nohup.out 2>&1 &
# nohup python cdan_mcc_sdat.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 30 --seed 0 -b 32 --log logs/cdan_mcc_sdat_resnet50/OfficeHome_Rw2Pr --log_name Rw2Pr_cdan_mcc_sdat_resnet50 --gpu 0 --rho 0.02 --lr 0.01 --log_results  > nohup.out 2>&1 &

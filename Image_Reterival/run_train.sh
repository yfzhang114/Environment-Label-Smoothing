CUDA_VISIBLE_DEVICES=1 nohup python3 ./tools/train_net.py --config-file configs/Base-DANN.yml --tsne-only >dann-tsne.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python3 ./tools/train_net.py --config-file /data2/User/DGWD/configs/Sample/Baseline-mobilenet.yml >baseline.out 2>&1 &
# conda install pytorch=1.6 torchvision cudatoolkit=10.1 -c pytorch 
# pip install tensorboard
# pip install Cython
# pip install yacs
# pip install termcolor
# pip install tabulate
# pip install scikit-learn

# pip install h5py
# pip install imageio
# pip install openpyxl 
# pip install matplotlib 
# pip install pandas 
# pip install seaborn
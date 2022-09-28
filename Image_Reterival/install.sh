# echo y| conda create -n dgwd python=3.6
# conda activate dgwd
echo y| conda install pytorch=1.6 torchvision cudatoolkit=10.1 -c pytorch 
pip install tensorboard
pip install Cython
pip install yacs
pip install termcolor
pip install tabulate
pip install scikit-learn

pip install h5py
pip install imageio
pip install openpyxl 
pip install matplotlib 
pip install pandas 
pip install seaborn
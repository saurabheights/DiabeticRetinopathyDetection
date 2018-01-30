

#https://gist.github.com/bzamecnik/b0c342d22a2a21f6af9d10eba3d4597b

#new cuda version needed 


#install driver: 

wget http://us.download.nvidia.com/tesla/390.12/nvidia-diag-driver-local-repo-ubuntu1604-390.12_1.0-1_amd64.deb
sudo apt-key add /var/nvidia-diag-driver-local-repo-390.12/7fa2af80.pub
sudo dpkg -i nvidia-diag-driver-local-repo-ubuntu1604-390.12_1.0-1_amd64.deb 
sudo apt-get update
sudo apt-get install cuda-drivers
sudo reboot


## step 2 
nvidia-smi
echo "Check output of nvidia-smi"

wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
sudo reboot 


#afterwards export of path needed: 




curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
echo "Installing miniconda in default location ~/miniconda3" 
bash ./Miniconda3-latest-Linux-x86_64.sh -b

echo "Adding conda and cuda in PATH in bashrc"

export PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}} >> ~/.bashrc

export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64{LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}  >> ~/.bashrc

echo 'export PATH=~/miniconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

conda install -y numpy=1.14.0 opencv=3.3.1 
conda install -y -c anaconda pillow=5.0.0 pandas=0.22.0 jupyter=1.0.0
conda install -y -c conda-forge matplotlib=2.1.1 cycler=0.10.0 progressbar2=3.34.3
conda install pytorch 


git clone https://github.com/saurabheights/DiabeticRetinopathyDetection/

git clone https://github.com/facebookresearch/SparseConvNet

cd SparseConvNet/PyTorch
sudo apt-get install libsparsehash-dev
sudo apt-get install unrar 
pip install git+https://github.com/pytorch/tnt.git@master
python setup.py develop





#https://gist.github.com/bzamecnik/b0c342d22a2a21f6af9d10eba3d4597b

#new cuda version needed 


#install driver: 

wget http://us.download.nvidia.com/tesla/390.12/nvidia-diag-driver-local-repo-ubuntu1604-390.12_1.0-1_amd64.deb
sudo apt-key add /var/nvidia-diag-driver-local-repo-390.12/7fa2af80.pub
sudo dpkg -i nvidia-diag-driver-local-repo-ubuntu1604-390.12_1.0-1_amd64.deb 
sudo apt-get update
sudo apt-get install cuda-drivers
sudo reboot


## step 2 
nvidia-smi
echo "Check output of nvidia-smi"

wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
sudo reboot 


#afterwards export of path needed: 




curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
echo "Installing miniconda in default location ~/miniconda3" 
bash ./Miniconda3-latest-Linux-x86_64.sh -b

echo "Adding conda and cuda in PATH in bashrc"

export PATH=/usr/local/cuda-9.1/bin${PATH:+:${PATH}} >> ~/.bashrc

export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64{LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}  >> ~/.bashrc

echo 'export PATH=~/miniconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

conda install -y numpy=1.14.0 opencv=3.3.1 
conda install -y -c anaconda pillow=5.0.0 pandas=0.22.0 jupyter=1.0.0
conda install -y -c conda-forge matplotlib=2.1.1 cycler=0.10.0 progressbar2=3.34.3
conda install pytorch 


git clone https://github.com/saurabheights/DiabeticRetinopathyDetection/

git clone https://github.com/facebookresearch/SparseConvNet

cd SparseConvNet/PyTorch
sudo apt-get install libsparsehash-dev
sudo apt-get install unrar 
pip install git+https://github.com/pytorch/tnt.git@master
python setup.py develop





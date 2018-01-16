#Install Miniconda - Faster and precise - Python 3.6.3
curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
 
echo "Installing miniconda in default location /home/<USER>/miniconda3" 
bash ./Miniconda3-latest-Linux-x86_64.sh -b

echo "Installing dependencies - numpy, pillow, pandas, torch and torchvision" 
conda install -y numpy=1.14.0
conda install -y -c anaconda pillow=5.0.0
conda install -y -c anaconda pandas=0.22.0
# Use channel pytorch to install right version of pytorch and torchvision
conda install -y pytorch=0.3.0 torchvision=0.2.0 -c pytorch
conda install -y opencv=3.3.1
conda install -y -c anaconda jupyter 




# Script to create virtual environment with Anaconda and install needed packages
conda create --name cxr python=3.9 -y
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cxr
conda install pip -y
pip install -r requirements.txt

# Install yolov5
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

# # Change directory
cd ~

# Install nnUNetv1
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
git checkout nnunetv1
pip install -e .

# nnUNetv1 paths
export nnUNet_raw_data_base="$HOME/media/nnUNet_raw_data_base"
export nnUNet_preprocessed="$HOME/media/nnUNet_preprocessed"
export RESULTS_FOLDER="$HOME/media/nnUNet_results"

# Print env vars
echo "nnUNet paths:"
echo ${nnUNet_raw_data_base} 
echo ${nnUNet_preprocessed} 
echo ${RESULTS_FOLDER}

# Change directory
cd ~

# Paths to .bashrc
echo "export nnUNet_raw_data_base=$nnUNet_raw_data_base" >> ~/.bashrc
echo "export nnUNet_preprocessed=$nnUNet_preprocessed" >> ~/.bashrc
echo "export RESULTS_FOLDER=$RESULTS_FOLDER" >> ~/.bashrc

# Execute .bashrc
source ~/.bashrc
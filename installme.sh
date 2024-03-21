#!/bin/bash

# Environment name variable - change "custom-env-name" to your desired environment name
ENV_NAME="py-BB"

# Installing conda
mkdir ./Executables
if [ "$(uname)" == "Darwin" ]; then
    # Do something under Mac OS X platform
    if [ "$(uname -m)" == "x86_64" ]; then
        wget -O ./Executables/Miniforge3-latest.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
    elif [ "$(uname -m)" == "arm64" ]; then
        wget -O ./Executables/Miniforge3-latest.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
    fi
elif [ "$(uname)" == "Linux" ]; then
    # Do something under Linux platform
    wget -O ./Executables/Miniforge3-latest.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
fi
bash ./Executables/Miniforge3-latest.sh -b -p ./Executables/miniforge3 -f

# Create your own virtual environment in a new folder
source ./Executables/miniforge3/bin/activate

conda update -n base -c conda-forge conda
conda create -n $ENV_NAME python=3.10
conda activate $ENV_NAME

# FOR GPU Calculations
#=======================================
conda install mamba -n base -c conda-forge
pip install cupy-cuda11x
./Executables/miniforge3/bin/mamba install cudatoolkit=11.8.0
pip install gpustat 

# Install generic python packages
#========================================
pip install jupyterlab
pip install ipywidgets
pip install PyYAML
pip install pyarrow
pip install pandas
pip install dask
pip install bokeh
pip install matplotlib
pip install scipy
pip install ipympl
pip install ruamel.yaml
pip install rich
pip install lfm
pip install pytest
pip install twine

# Adding the jupyter kernel to the list of kernels
python -m ipykernel install --user --name $ENV_NAME --display-name "$ENV_NAME"

# Install project package
pip install -e ./

# Install CERN packages
#=========================================
# Installing fortran and other compilers
conda install compilers cmake

pip install cpymad

git clone https://github.com/pbelange/nafflib.git ./Executables/$ENV_NAME/nafflib
pip install -e ./Executables/$ENV_NAME/nafflib

git clone https://github.com/lhcopt/lhcmask.git ./Executables/$ENV_NAME/lhcmask
pip install -e ./Executables/$ENV_NAME/lhcmask

git clone https://github.com/xsuite/xobjects ./Executables/$ENV_NAME/xobjects
pip install -e ./Executables/$ENV_NAME/xobjects

git clone https://github.com/xsuite/xdeps ./Executables/$ENV_NAME/xdeps
pip install -e ./Executables/$ENV_NAME/xdeps

git clone https://github.com/xsuite/xpart ./Executables/$ENV_NAME/xpart
pip install -e ./Executables/$ENV_NAME/xpart

git clone https://github.com/xsuite/xtrack ./Executables/$ENV_NAME/xtrack
pip install -e ./Executables/$ENV_NAME/xtrack

git clone https://github.com/xsuite/xmask ./Executables/$ENV_NAME/xmask
pip install -e ./Executables/$ENV_NAME/xmask

git clone https://github.com/xsuite/xfields ./Executables/$ENV_NAME/xfields
pip install -e ./Executables/$ENV_NAME/xfields

git clone https://github.com/PyCOMPLETE/FillingPatterns.git ./Executables/$ENV_NAME/FillingPatterns
pip install -e ./Executables/$ENV_NAME/FillingPatterns

git clone https://gitlab.cern.ch/mrufolo/fillingstudies.git ./Executables/$ENV_NAME/fillingstudies
pip install -e ./Executables/$ENV_NAME/fillingstudies

# Download outsourced files
#=========================================
cd ./Executables/$ENV_NAME/xmask
git submodule init
git submodule update

# Downloading sequences
cd ../../../
wget https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/2023/lhc.seq -P Machines/sequences/
wget https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/2023/lhcb4.seq -P Machines/sequences/

# Downloading macro file
wget https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/2023/toolkit/macro.madx -P Machines/toolkit/
wget https://gitlab.cern.ch/acc-models/acc-models-lhc/-/blob/2023/toolkit/slice.madx -P Machines/toolkit/

# Downloading optics
rsync -rv phbelang@lxplus.cern.ch:"/afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/Proton_2022/opticsfile.*" ./Machines/optics/
rsync -rv phbelang@lxplus.cern.ch:"/afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/Proton_2022/README" ./Machines/optics/

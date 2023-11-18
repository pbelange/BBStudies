                                              

# installing conda
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
    wget -O ./Executables/Miniforge3-latest.sh ./Executables/Miniconda3-latest.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
fi
bash ./Executables/Miniforge3-latest.sh -b  -p ./Executables/miniforge3 -f


# create your own virtual environment in a new folder
source ./Executables/miniforge3/bin/activate

conda update -n base -c conda-forge conda
conda create -n py-BB python=3.10
conda activate py-BB

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
pip install pynaff
pip install NAFFlib

# Adding the jupyter kernel to the list of kernels
python -m ipykernel install --user --name py-BB --display-name "py-BB"
# ========================================


# Install project package
pip install -e ./

# Install CERN packages
#=========================================
# Installing fortran an other compilers
conda install compilers cmake

pip install cpymad

git clone https://github.com/lhcopt/lhcmask.git ./Executables/py-BB/lhcmask
pip install -e ./Executables/py-BB/lhcmask

git clone https://github.com/xsuite/xobjects ./Executables/py-BB/xobjects
pip install -e ./Executables/py-BB/xobjects

git clone https://github.com/xsuite/xdeps ./Executables/py-BB/xdeps
pip install -e ./Executables/py-BB/xdeps

git clone https://github.com/xsuite/xpart ./Executables/py-BB/xpart
pip install -e ./Executables/py-BB/xpart

git clone https://github.com/xsuite/xtrack ./Executables/py-BB/xtrack
pip install -e ./Executables/py-BB/xtrack

git clone https://github.com/xsuite/xmask ./Executables/py-BB/xmask
pip install -e ./Executables/py-BB/xmask

git clone https://github.com/xsuite/xfields ./Executables/py-BB/xfields
pip install -e ./Executables/py-BB/xfields

git clone https://github.com/PyCOMPLETE/FillingPatterns.git ./Executables/py-BB/FillingPatterns
pip install -e ./Executables/py-BB/FillingPatterns

git clone https://gitlab.cern.ch/mrufolo/fillingstudies.git ./Executables/py-BB/fillingstudies
pip install -e ./Executables/py-BB/fillingstudies
#=========================================



# Download outsourced files
#=========================================
cd ./Executables/py-BB/xmask
git submodule init
git submodule update

# Downloading sequences
cd ../../../
wget https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/2023/lhc.seq -P Machines/Sequences/
wget https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/2023/lhcb4.seq -P Machines/Sequences/

# Downloading macro file
wget https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/2023/toolkit/macro.madx -P Machines/Toolkit/
wget https://gitlab.cern.ch/acc-models/acc-models-lhc/-/blob/2023/toolkit/slice.madx -P Machines/Toolkit/


# Downloading optics
rsync -rv phbelang@lxplus.cern.ch:"/afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/2021_V6/PROTON/opticsfile.* /afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/2021_V6/PROTON/README" ./Machines/Optics/
#=========================================
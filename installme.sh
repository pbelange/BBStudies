                                              

# installing conda
mkdir ./Executables
if [ "$(uname)" == "Darwin" ]; then
    # Do something under Mac OS X platform
    if [ "$(uname -m)" == "x86_64" ]; then
        wget -O ./Executables/Miniconda3-latest.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    elif [ "$(uname -m)" == "arm64" ]; then
        wget -O ./Executables/Miniconda3-latest.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
    fi
elif [ "$(uname)" == "Linux" ]; then
    # Do something under Linux platform
    wget -O ./Executables/Miniconda3-latest.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi
bash ./Executables/Miniconda3-latest.sh -b  -p ./Executables/miniconda -f


# create your own virtual environment in a new folder
source ./Executables/miniconda/bin/activate
python -m venv ./Executables/py_BB
source ./Executables/py_BB/bin/activate


# Install generic python packages
#========================================
pip install jupyterlab
pip install ipywidgets
pip install PyYAML
pip install pyarrow
pip install pandas
pip install matplotlib
pip install scipy
pip install ipympl
pip install ruamel.yaml
pip install rich
pip install lfm
pip install pynaff
pip install NAFFlib

# Adding the jupyter kernel to the list of kernels
python -m ipykernel install --user --name py_BB --display-name "py_BB"
#========================================


# Install CERN packages
#=========================================
pip install cpymad

git clone https://github.com/lhcopt/lhcmask.git ./Executables/py_BB/lhcmask
pip install -e ./Executables/py_BB/lhcmask

git clone https://github.com/xsuite/xobjects ./Executables/py_BB/xobjects
pip install -e ./Executables/py_BB/xobjects

git clone https://github.com/xsuite/xdeps ./Executables/py_BB/xdeps
pip install -e ./Executables/py_BB/xdeps

git clone https://github.com/xsuite/xpart ./Executables/py_BB/xpart
pip install -e ./Executables/py_BB/xpart

git clone https://github.com/xsuite/xtrack ./Executables/py_BB/xtrack
pip install -e ./Executables/py_BB/xtrack

git clone https://github.com/xsuite/xmask ./Executables/py_BB/xmask
pip install -e ./Executables/py_BB/xmask

git clone https://github.com/xsuite/xfields ./Executables/py_BB/xfields
pip install -e ./Executables/py_BB/xfields

git clone https://github.com/PyCOMPLETE/FillingPatterns.git ./Executables/py_BB/FillingPatterns
pip install -e ./Executables/py_BB/FillingPatterns

git clone https://gitlab.cern.ch/mrufolo/fillingstudies.git ./Executables/py_BB/fillingstudies
#pip install -e ./Executables/py_wireDAQ/fillingstudies
#=========================================
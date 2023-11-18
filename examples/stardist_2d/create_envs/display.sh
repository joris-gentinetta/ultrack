conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda create -n display python=3.10 -y
conda activate display || exit
conda install -c conda-forge napari pyqt
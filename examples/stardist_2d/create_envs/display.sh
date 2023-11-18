
conda env remove -p /cluster/scratch/jorisg/conda/display -y
conda create -n display python=3.10 -y
conda activate display || exit
conda install -c conda-forge napari pyqt
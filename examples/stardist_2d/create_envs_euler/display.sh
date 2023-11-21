
conda env remove -p /cluster/scratch/jorisg/conda/display -y
conda create --prefix /cluster/scratch/jorisg/conda/ python=3.10 -y
conda activate -p /cluster/scratch/jorisg/conda/ display || exit
conda install -c conda-forge napari pyqt
conda install matplotlib
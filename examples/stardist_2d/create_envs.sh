conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda create -n track python=3.10 -y
conda activate track || exit
pip install ultrack
conda install pytorch cpuonly -c pytorch -y
#conda install -c gurobi gurobi #todo license

conda create -n segment_stardist python=3.10 -y
conda activate segment_stardist || exit
pip install tensorflow
pip install stardist

conda create -n display python=3.10 -y
conda activate display || exit
conda install -c conda-forge napari pyqt
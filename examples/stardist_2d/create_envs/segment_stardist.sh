conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda create -n segment_stardist python=3.10 -y
conda activate segment_stardist || exit
pip install tensorflow
pip install stardist

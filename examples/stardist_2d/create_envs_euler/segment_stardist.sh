conda env remove -p /cluster/scratch/jorisg/conda/segment_stardist -y
conda create --prefix /cluster/scratch/jorisg/conda/segment_stardist python=3.10 -y
conda activate /cluster/scratch/jorisg/conda/segment_stardist || exit
pip install tensorflow
pip install stardist

conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda create --prefix /cluster/scratch/jorisg/conda/track python=3.10 -y
conda activate /cluster/scratch/jorisg/conda/track || exit
cd ../../../.. || exit
pip install ultrack
#conda install pytorch cpuonly -c pytorch -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
#conda install -c gurobi gurobi #todo license
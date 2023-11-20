
conda env remove -n track -y
conda create -n track python=3.10 -y
conda activate track || exit
cd ../../../.. || exit
pip install ultrack
conda install pytorch cpuonly -c pytorch -y
#conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
#conda install -c gurobi gurobi #todo license
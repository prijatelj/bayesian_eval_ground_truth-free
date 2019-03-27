source activate metric_py3

export ROOT=$PWD
export PYTHONPATH=$PWD:$PYTHONPATH
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export LIBRARY_PATH="$LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda-9.2/
export PATH=/usr/local/cuda-9.2/bin:$PATH

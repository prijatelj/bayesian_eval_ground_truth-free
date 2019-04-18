export ROOT=$PWD
export PYTHONPATH=$PWD:$PYTHONPATH
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export LIBRARY_PATH="$LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda-10.0/
export PATH=/usr/local/cuda-10.0/bin:$PATH

alias metric="cd $ROOT"

echo "source activate metric_py3 && conda deactivate && conda activate metric_py3"

CUDA_VISIBLE_DEVICES=$1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
set -e 
set -u
export CUDA_VISIBLE_DEVICES=$1
target=$2
#~/Python/bin/python3 -u train_batch.py $2
~/Python_cpu/bin/python3 -u train_batch.py $2

#!/bin/bash
#SBATCH --job-name=poc_bleurt_down_17_       # 任务名
#SBATCH --nodes=1                   # 这里不用动 多节点脚本请查官方文档
#SBATCH --ntasks=1                  # 这里不用动 多任务脚本请查官方文档
#SBATCH --cpus-per-task=4           # 要几块CPU (一般4块就够用了)
#SBATCH --mem=64GB                 # 最大内存
#SBATCH --time=12:00:00           # 运行时间上限
#SBATCH --mail-type=END             # ALL / END
#SBATCH --mail-user=yh2689@nyu.edu  # 结束之后给哪里发邮件
#SBATCH --output=./logs/poc/%x%A.out           # 正常输出写入的文件
#SBATCH --error=./logs/poc/%x%A.err            # 报错信息写入的文件
#SBATCH -q cpu-512                  # 有GPU的partition

nvidia-smi
nvcc --version
cd /l/users/yichen.huang/eval_attack/code   # 切到程序目录

echo "START"               # 输出起始信息
source /apps/local/anaconda3/bin/activate adv          # 调用 virtual env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export HF_HOME=/l/users/yichen.huang/misc/cache
python -u poc.py \
    --name poc \
    --goal_direction down \
    --victim bleurt \
    --goal_abs_delta 0.2 \
    --n_samples 100 \
    --dataset 2017-da \
    --log_prob_diff 3
python -u poc.py \
    --name poc \
    --victim bleurt \
    --goal_direction down \
    --goal_abs_delta 0.35 \
    --n_samples 100 \
    --dataset 2017-da \
    --log_prob_diff 3
echo "FINISH"                       # 输出起始信息

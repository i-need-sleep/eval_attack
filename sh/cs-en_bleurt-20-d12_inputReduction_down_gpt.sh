#!/bin/bash
#SBATCH --job-name=cs-en_bleurt-20-d12_inputReduction_down_gpt       # 任务名
#SBATCH --nodes=1                   # 这里不用动 多节点脚本请查官方文档
#SBATCH --ntasks=1                  # 这里不用动 多任务脚本请查官方文档
#SBATCH --cpus-per-task=4           # 要几块CPU (一般4块就够用了)
#SBATCH --mem=256GB                 # 最大内存
#SBATCH --time=12:00:00           # 运行时间上限
#SBATCH --mail-type=END             # ALL / END
#SBATCH --mail-user=yh2689@nyu.edu  # 结束之后给哪里发邮件
#SBATCH --output=./logs/cs-en/%x%A.out           # 正常输出写入的文件
#SBATCH --error=./logs/cs-en/%x%A.err            # 报错信息写入的文件
#SBATCH --gres=gpu:1                # 需要几块GPU (同时最多8块)
#SBATCH -p gpu                   # 有GPU的partition
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs

nvidia-smi
nvcc --version
cd /l/users/yichen.huang/eval_attack/code   # 切到程序目录

echo "START"               # 输出起始信息
source /apps/local/anaconda3/bin/activate adv          # 调用 virtual env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export HF_HOME=/l/users/yichen.huang/misc/cache
python -u poc.py \
    --name 20-d12 \
    --dataset aggregated_cs-en_bleurt-20-d12 \
    --use_normalized \
    --victim bleurt \
    --bleurt_checkpoint bleurt-20-d12 \
    --goal_direction down \
    --goal_abs_delta 1 \
    --attack_algo input_reduction
echo "FINISH"                       # 输出起始信息
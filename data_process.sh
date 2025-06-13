#!/bin/bash
#SBATCH --job-name=nuplan   # 作业名，可选
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16           # 申请调用CPU线程数
#SBATCH --nodelist=3090node2
#SBATCH --output=log/output.log              # 标准输出文件
#SBATCH --error=log/error.log                # 错误输出文件
###################################
# User Configuration Section
###################################

# NUPLAN_DATA_PATH="/mnt/slurmfs-A100/user_data/ryao092/datasets/nuplan/" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
# NUPLAN_MAP_PATH= "/mnt/slurmfs-A100/user_data/ryao092/datasets/nuplan/maps" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")
source /mnt/slurmfs-4090node1/homes/rzhong151/anaconda3/bin/activate diffusion_planner

TRAIN_SET_PATH="TRAIN_SET_PATH" # preprocess training data
###################################

python data_process.py \
--save_path $TRAIN_SET_PATH \
--total_scenarios 1000000 \


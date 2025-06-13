#!/bin/bash
#SBATCH --job-name=acctrajectory_planning   # 作业名，可选
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16           # 申请调用CPU线程数
#SBATCH --nodelist=3090node1
#SBATCH --output=acc/output.log              # 标准输出文件
#SBATCH --error=acc/error.log                # 错误输出文件

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="REPLACE_WITH_PYTHON_PATH" # python path (e.g., "/home/xxx/anaconda3/envs/diffusion_planner/bin/python")


# Set training data path
# TRAIN_SET_PATH="REPLACE_WITH_TRAIN_SET_PATH" # preprocess data using data_process.sh
# TRAIN_SET_LIST_PATH="REPLACE_WITH_TRAIN_SET_LIST_PATH"
###################################

sudo -E $RUN_PYTHON_PATH -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
# --train_set  $TRAIN_SET_PATH \
# --train_set_list  $TRAIN_SET_LIST_PATH \


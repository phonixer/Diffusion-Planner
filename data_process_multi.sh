#!/bin/bash
#SBATCH --job-name=nuplan   # 作业名，可选
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16           # 申请调用CPU线程数
#SBATCH --nodelist=3090node3
#SBATCH --output=log/output.log              # 标准输出文件
#SBATCH --error=log/error.log                # 错误输出文件
###################################
# User Configuration Section
###################################

# NUPLAN_DATA_PATH="/mnt/slurmfs-A100/user_data/ryao092/datasets/nuplan/" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
# NUPLAN_MAP_PATH= "/mnt/slurmfs-A100/user_data/ryao092/datasets/nuplan/maps" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")
source /mnt/slurmfs-4090node1/homes/rzhong151/anaconda3/bin/activate diffusion_planner

TRAIN_SET_PATH="/mnt/slurmfs-4090node2/user_data/mpeng060/diffusion_planner/train_data" # preprocess training data
###################################

TOTAL_SCENARIOS=1000000
NUM_PROCESSES=80

# Calculate scenarios per process
SCENARIOS_PER_PROCESS=$((TOTAL_SCENARIOS / NUM_PROCESSES))

# Create log directory
mkdir -p log

# Run multiple processes in parallel
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    START_INDEX=$((i * SCENARIOS_PER_PROCESS))
    END_INDEX=$((START_INDEX + SCENARIOS_PER_PROCESS))
    
    # Run each process with its own log files
    python data_process.py \
        --save_path $TRAIN_SET_PATH \
        --total_scenarios $SCENARIOS_PER_PROCESS \
        --start_index $START_INDEX \
        --end_index $END_INDEX \
        --checkpoint_interval 1000 \
        > log/process_${i}.log 2>&1 &
done

# Wait for all processes to complete
wait

echo "All processes completed"


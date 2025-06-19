import os
import argparse
import json

from diffusion_planner.data_process.data_processor import DataProcessor

import time
import random
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping

def get_filter_parameters(num_scenarios_per_type=None, limit_total_scenarios=None, shuffle=True, scenario_tokens=None, log_names=None):

    scenario_types = None

    scenario_tokens                      # List of scenario tokens to include
    log_names = log_names                # Filter scenarios by log names
    map_names = None                     # Filter scenarios by map names

    num_scenarios_per_type               # Number of scenarios per type
    limit_total_scenarios                # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
    timestamp_threshold_s = None         # Filter scenarios to ensure scenarios have more than timestamp_threshold_s seconds between their initial lidar timestamps
    ego_displacement_minimum_m = None    # Whether to remove scenarios where the ego moves less than a certain amount

    expand_scenarios = True              # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = False          # Whether to remove scenarios where the mission goal is invalid
    shuffle                              # Whether to shuffle the scenarios

    ego_start_speed_threshold = None     # Limit to scenarios where the ego reaches a certain speed from below
    ego_stop_speed_threshold = None      # Limit to scenarios where the ego reaches a certain speed from above
    speed_noise_tolerance = None         # Value at or below which a speed change between two timepoints should be ignored as noise.

    return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, ego_displacement_minimum_m, \
           expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance


from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import multiprocessing
import gc
import psutil
def process_chunk(scenarios_chunk, config):
    """处理场景块，包含错误处理和内存管理"""
    try:
        # 每个子进程重新创建一个 DataProcessor 实例
        processor = DataProcessor(config)
        
        # 分批处理，避免内存问题
        batch_size = 64  # 每批处理5个场景
        processed_count = 0
        
        for i in range(0, len(scenarios_chunk), batch_size):
            if i + batch_size > len(scenarios_chunk):
                # 如果剩余的场景不足一批，调整批大小
                batch_size = len(scenarios_chunk) - i
            if i % 10 == 0:
                print(f"处理场景 {i} 到 {i + batch_size}，总计 {len(scenarios_chunk)} 个场景")
            batch = scenarios_chunk[i:i+batch_size]
            processor.work(batch)
            processed_count += len(batch)
            
            if i % (batch_size * 5) == 0:  # 每处理约320个场景后收一次
                gc.collect()

            
            # 检查内存使用情况
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > 32000:  # 超过4GB发出警告
                print(f"警告: 进程内存使用: {memory_mb:.1f}MB")
        
        return processed_count
        
    except Exception as e:
        print(f"处理块时出错: {e}")
        return -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--data_path', default='/mnt/slurmfs-A100/user_data/ryao092/datasets/nuplan/nuplan-v1.1/splits/trainval', type=str, help='path to raw data')
    parser.add_argument('--map_path', default='/mnt/slurmfs-A100/user_data/ryao092/datasets/nuplan/maps', type=str, help='path to map data')

    parser.add_argument('--save_path', default='./TRAIN_SET_PATH', type=str, help='path to save processed data')
    parser.add_argument('--scenarios_per_type', type=int, default=None, help='number of scenarios per type')
    parser.add_argument('--total_scenarios', type=int, default=1000000, help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios', type=bool, default=True, help='shuffle scenarios')

    parser.add_argument('--agent_num', type=int, help='number of agents', default=32)
    parser.add_argument('--static_objects_num', type=int, help='number of static objects', default=5)

    parser.add_argument('--lane_len', type=int, help='number of lane point', default=20)
    parser.add_argument('--lane_num', type=int, help='number of lanes', default=70)

    parser.add_argument('--route_len', type=int, help='number of route lane point', default=20)
    parser.add_argument('--route_num', type=int, help='number of route lanes', default=25)
    # nuplan_train
    parser.add_argument('--nuplan_train', type=str, default='./nuplan_train_copy.json', help='path to nuplan train json file')
    args = parser.parse_args()

    # create save folder
    os.makedirs(args.save_path, exist_ok=True)
    print(f"Save path: {args.save_path}")
    import multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() - 16)
    print(f"Number of workers: {num_workers}")
    sensor_root = None
    db_files = None

    # Only preprocess the training data
    with open(args.nuplan_train, 'r') as file:
        log_names = json.load(file)

    # 把这个分成多个batch来处理
    # 例如每个batch处理1000个场景
    # 这样可以避免内存问题
    # 这里的log_names是一个列表，包含了所有需要处理的日志名称
    



    map_version = "nuplan-maps-v1.0"    
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version)
    scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type, args.total_scenarios, args.shuffle_scenarios, log_names=log_names))

    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")
    # 从 第442833 之后开始
    # Process subset of scenarios based on start/end index
    # args.start_index = 442833 - 5
    # args.end_index = args.total_scenarios  # None means process all remaining scenarios
    # scenarios = scenarios[args.start_index:args.end_index]

    # # process data
    # del worker, builder, scenario_filter
    # processor = DataProcessor(args)
    # processor.work(scenarios)

    num_workers = num_workers  # 你机器上可用的核心数或 SLURM --ntasks-per-node
    
    # 把 scenarios 列表等分成 num_workers 份
    chunks = [
        scenarios[i::num_workers]
        for i in range(num_workers)
    ]

    def split_list_evenly(lst, num_chunks):
        """将列表尽可能平均分成num_chunks份"""
        chunk_size = len(lst) // num_chunks
        remainder = len(lst) % num_chunks
        
        chunks = []
        start = 0
        for i in range(num_chunks):
            # 前remainder个chunk多分配1个元素
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end = start + current_chunk_size
            chunks.append(lst[start:end])
            start = end
            
        return chunks

    chunks = split_list_evenly(scenarios, num_workers)
    print("分块情况:")
    for i, chunk in enumerate(chunks):
        print(f"Worker {i}: {len(chunk)} scenarios")
    print(f"总计: {sum(len(chunk) for chunk in chunks)} scenarios")

    # 使用更健壮的处理方式
    processed_total = 0
    failed_chunks = []
    
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as exe:
            # 提交所有任务
            future_to_chunk = {exe.submit(process_chunk, chunk, args): i for i, chunk in enumerate(chunks)}
            
            # 处理完成的结果
            from concurrent.futures import as_completed
            
            for future in tqdm(as_completed(future_to_chunk), total=len(chunks), desc="处理进度"):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result(timeout=180)  # 30分钟超时
                    if result > 0:
                        processed_total += result
                        print(f"块 {chunk_idx} 完成: {result} 个场景")
                    else:
                        failed_chunks.append(chunk_idx)
                        print(f"块 {chunk_idx} 处理失败")
                except Exception as e:
                    print(f"块 {chunk_idx} 出错: {e}")
                    failed_chunks.append(chunk_idx)
    
    except Exception as e:
        print(f"多进程执行出错: {e}")
        print("切换到顺序处理...")
        
        # 降级到顺序处理
        processor = DataProcessor(args)
        for i, chunk in enumerate(tqdm(chunks, desc="顺序处理")):
            try:
                processor.work(chunk)
                processed_total += len(chunk)
                print(f"顺序处理块 {i} 完成: {len(chunk)} 个场景")
                
                # 定期清理内存
                if i % 2 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"顺序处理块 {i} 失败: {e}")
                failed_chunks.append(i)
    
    print(f"处理完成。总共处理: {processed_total} 个场景")
    if failed_chunks:
        print(f"失败的块: {failed_chunks}")

    npz_files = [f for f in os.listdir(args.save_path) if f.endswith('.npz')]

    # 保存文件列表到JSON
    with open('./diffusion_planner_training.json', 'w') as json_file:
        json.dump(npz_files, json_file, indent=4)

    print(f"保存了 {len(npz_files)} 个 .npz 文件名")
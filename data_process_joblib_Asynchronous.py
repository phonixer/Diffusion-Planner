import os
import argparse
import json
import time
import gc
import psutil
import random
import multiprocessing
import threading  # 需添加到文件开头的导入部分
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import argparse
import json
import time
import gc
import psutil
import random
import multiprocessing
import threading
import asyncio
from tqdm import tqdm
from joblib import Parallel, delayed

from diffusion_planner.data_process.data_processor2 import AsyncDataProcessor
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping

def get_filter_parameters(num_scenarios_per_type=None, limit_total_scenarios=None, shuffle=True, scenario_tokens=None, log_names=None):
    scenario_types = None
    map_names = None
    timestamp_threshold_s = None
    ego_displacement_minimum_m = None
    expand_scenarios = True
    remove_invalid_goals = False
    ego_start_speed_threshold = None
    ego_stop_speed_threshold = None
    speed_noise_tolerance = None

    return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, \
           ego_displacement_minimum_m, expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, \
           ego_stop_speed_threshold, speed_noise_tolerance

def process_chunk_sync(scenarios_chunk, config):
    """同步处理函数，用于多进程环境"""
    print(
        f"Process ID: {os.getpid()}, "
        f"Thread ID: {threading.get_ident()}, "
        f"Is main thread: {threading.current_thread() is threading.main_thread()}"
    )
    
    # 在每个进程中创建独立的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        processor = AsyncDataProcessor(config)
        processed_count = loop.run_until_complete(
            process_chunk_async(processor, scenarios_chunk))
        return processed_count
    except Exception as e:
        print(f"Chunk processing error: {e}")
        return -1
    finally:
        loop.close()

async def process_chunk_async(processor, scenarios_chunk):
    """实际的异步处理逻辑"""
    await processor.start()
    processed_count = 0
    batch_size = 64

    try:
        for i in range(0, len(scenarios_chunk), batch_size):
            batch = scenarios_chunk[i:i + batch_size]
            
            # 使用asyncio.gather并行处理批次
            tasks = [processor.process_scenario_async(scenario) 
                    for scenario in batch]
            results = await asyncio.gather(*tasks)
            
            processed_count += sum(results)
            
            # 内存管理
            if i % 2 == 0:
                gc.collect()
            
            # 内存监控
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            if memory_mb > 32000:
                print(f"WARNING: Memory usage high: {memory_mb:.1f}MB")
                
        return processed_count
    finally:
        await processor.stop()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--data_path', type=str, default='/mnt/slurmfs-A100/user_data/ryao092/datasets/nuplan/nuplan-v1.1/splits/trainval')
    parser.add_argument('--map_path', type=str, default='/mnt/slurmfs-A100/user_data/ryao092/datasets/nuplan/maps')
    parser.add_argument('--save_path', type=str, default='./TRAIN_SET_PATH')
    parser.add_argument('--scenarios_per_type', type=int, default=None)
    parser.add_argument('--total_scenarios', type=int, default=1000000)
    parser.add_argument('--shuffle_scenarios', type=bool, default=True)
    parser.add_argument('--agent_num', type=int, default=32)
    parser.add_argument('--static_objects_num', type=int, default=5)
    parser.add_argument('--lane_len', type=int, default=20)
    parser.add_argument('--lane_num', type=int, default=70)
    parser.add_argument('--route_len', type=int, default=20)
    parser.add_argument('--route_num', type=int, default=25)
    parser.add_argument('--nuplan_train', type=str, default='./nuplan_train_json/nuplan_train_chunk_10.json')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    print(f"Save path: {args.save_path}")

    num_workers = max(1, multiprocessing.cpu_count() - 32)
    print(f"Number of workers: {num_workers}")

    with open(args.nuplan_train, 'r') as file:
        log_names = json.load(file)

    map_version = "nuplan-maps-v1.0"
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, None, None, map_version)
    scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type, args.total_scenarios, args.shuffle_scenarios, log_names=log_names))
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")

    def split_list_evenly(lst, num_chunks):
        chunk_size = len(lst) // num_chunks
        remainder = len(lst) % num_chunks
        chunks = []
        start = 0
        for i in range(num_chunks):
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end = start + current_chunk_size
            chunks.append(lst[start:end])
            start = end
        return chunks

    chunks = split_list_evenly(scenarios, num_workers)
    print("Chunk distribution:")
    for i, chunk in enumerate(chunks):
        print(f"Worker {i}: {len(chunk)} scenarios")
    print(f"Total: {sum(len(chunk) for chunk in chunks)} scenarios")

    print("\n>>> Starting parallel processing with joblib")
    start_time = time.time()

    # results = Parallel(n_jobs=num_workers, backend='loky', verbose=10)(
    #     delayed(process_chunk)(chunk, args) for chunk in chunks
    # )


    results = Parallel(n_jobs=num_workers, backend='loky')(
        delayed(process_chunk_sync)(chunk, args) for chunk in chunks
    )
#     results = Parallel(n_jobs=num_workers, backend='threading', batch_size='auto', prefer='threads')(
#     delayed(process_chunk)(chunk, args) for chunk in chunks
# )

    processed_total = sum(r for r in results if r > 0)
    failed_chunks = [i for i, r in enumerate(results) if r <= 0]

    print(f"\n✅ Finished: {processed_total} scenarios processed")
    if failed_chunks:
        print(f"❌ Failed chunks: {failed_chunks}")

    print(f"⏱️ Total processing time: {time.time() - start_time:.2f}s")

    npz_files = [f for f in os.listdir(args.save_path) if f.endswith('.npz')]
    with open('./diffusion_planner_training.json', 'w') as json_file:
        json.dump(npz_files, json_file, indent=4)
    print(f"📦 Saved {len(npz_files)} .npz filenames")

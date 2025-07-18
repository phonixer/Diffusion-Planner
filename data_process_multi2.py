import os
import argparse
import json

from diffusion_planner.data_process.data_processor import DataProcessor

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
    timestamp_threshold_s = None         # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--data_path', default='/mnt/slurmfs-A100/user_data/ryao092/datasets/nuplan/nuplan-v1.1/splits/trainval', type=str, help='path to raw data')
    parser.add_argument('--map_path', default='/mnt/slurmfs-A100/user_data/ryao092/datasets/nuplan/maps', type=str, help='path to map data')

    parser.add_argument('--save_path', default='./cache', type=str, help='path to save processed data')
    parser.add_argument('--scenarios_per_type', type=int, default=None, help='number of scenarios per type')
    parser.add_argument('--total_scenarios', type=int, default=10, help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios', type=bool, default=True, help='shuffle scenarios')

    parser.add_argument('--agent_num', type=int, help='number of agents', default=32)
    parser.add_argument('--static_objects_num', type=int, help='number of static objects', default=5)

    parser.add_argument('--lane_len', type=int, help='number of lane point', default=20)
    parser.add_argument('--lane_num', type=int, help='number of lanes', default=70)

    parser.add_argument('--route_len', type=int, help='number of route lane point', default=20)
    parser.add_argument('--route_num', type=int, help='number of route lanes', default=25)
        # Add new arguments for multi-processing
    parser.add_argument('--start_index', type=int, default=0, help='start index of scenarios')
    parser.add_argument('--end_index', type=int, default=None, help='end index of scenarios')
    parser.add_argument('--checkpoint_interval', type=int, default=100, help='interval to force gc collection')
    
    args = parser.parse_args()


    # create save folder
    os.makedirs(args.save_path, exist_ok=True)
    print(f"Save path: {args.save_path}")
    import multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() - 32)
    print(f"Number of workers: {num_workers}")
    sensor_root = None
    db_files = None

    # Only preprocess the training data
    with open('./nuplan_train.json', "r", encoding="utf-8") as file:
        log_names = json.load(file)

    map_version = "nuplan-maps-v1.0"    
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version)
    scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type, args.total_scenarios, args.shuffle_scenarios, log_names=log_names))

    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")



    # Process subset of scenarios based on start/end index
    if args.end_index is None:
        args.end_index = len(scenarios)
    scenarios_subset = scenarios[args.start_index:args.end_index]
    print(f"Processing scenarios from index {args.start_index} to {args.end_index}")

    # Process data
    processor = DataProcessor(args)
    processor.work(scenarios_subset)

    print(f"Processed {len(scenarios_subset)} scenarios")
    npz_files = [f for f in os.listdir(args.save_path) if f.endswith('.npz')]

    # 保存文件列表到JSON
    with open('./diffusion_planner_training.json', 'w') as json_file:
        json.dump(npz_files, json_file, indent=4)

    print(f"保存了 {len(npz_files)} 个 .npz 文件名")
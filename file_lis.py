import os
import json
import numpy as np
import time


save_path = "TRAIN_SET_PATH"

npz_files = [f for f in os.listdir(save_path) if f.endswith('.npz')]
# 保存文件列表到JSON
with open('./diffusion_planner_training_processed.json', 'w') as json_file:
    json.dump(npz_files, json_file, indent=4)

time1 = time.time()
filename = "diffusion_planner_training.json"
skipped_count = 0



print(f"Skipping already processed scenarios, total skipped: {skipped_count}")

time2 = time.time()
print(f"处理时间: {time2 - time1:.2f} 秒")
print(f"保存了 {len(npz_files)} 个 .npz 文件名")


# # 读取nuplan——train。json文件
# with open('./nuplan_train.json', 'r') as file:
#     nuplan_data = json.load(file)

# # 每100个文件保存一次
# chunk_size = 10
# chunks = [nuplan_data[i:i + chunk_size] for i in range(0, len(nuplan_data), chunk_size)]
# # 保存分块数据到新的JSON文件
# for i, chunk in enumerate(chunks):
#     with open(f'nuplan_train_json/nuplan_train_chunk_{i + 1}.json', 'w') as chunk_file:
#         json.dump(chunk, chunk_file, indent=4)
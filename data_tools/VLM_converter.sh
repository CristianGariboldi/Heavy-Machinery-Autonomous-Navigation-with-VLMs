#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/home/noah-22/QA_data_generation_CARLA:$PYTHONPATH

echo "Start generating dataset..."

# Process the entire dataset
/home/noah-22/miniconda3/envs/QArla/bin/python \
    data_tools/QA_data_generation.py \
    --data_path /home/noah-22/QA_data_generation_CARLA/data \
    --output_file /home/noah-22/QA_data_generation_CARLA/output \
    --max_frames 100000000000000000000000

# Alternatively, you can process only specific scenario types
# /home/noah-22/miniconda3/envs/QArla/bin/python \
#     data_tools/QA_data_generation.py \
#     --data_path /home/noah-22/QA_data_generation_CARLA/data \
#     --output_file /home/noah-22/QA_data_generation_CARLA/output/accident_dataset.json \
#     --process_type Accident \
#     --max_frames 100000000000000000000000

# Or process a single scenario folder
# /home/noah-22/miniconda3/envs/QArla/bin/python \
#     data_tools/QA_data_generation.py \
#     --data_path /home/noah-22/QA_data_generation_CARLA/data \
#     --output_file /home/noah-22/QA_data_generation_CARLA/output/single_scenario.json \
#     --process_single /home/noah-22/QA_data_generation_CARLA/data/Accident/Accident_1 \
#     --max_frames 100000000000000000000000

echo "Dataset generation complete."


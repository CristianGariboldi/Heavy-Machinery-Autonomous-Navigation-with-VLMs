#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

# Set the CUDA library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Force bitsandbytes to use the correct CUDA 11.8 libraries
export BNB_CUDA_VERSION=118

# Set the Python path for your project
export PYTHONPATH="/home/hestia-22/Desktop/Heavy-Machinery-Autonomous-Navigation-with-VLMs"

echo "Start generating dataset..."

/home/hestia-22/anaconda3/envs/sensmore/bin/python \
    data_tools/QA_data_generation.py \
    --image_folder ./video_frames_reduced \
    --output_dir ./output_data

echo "Dataset generation complete."
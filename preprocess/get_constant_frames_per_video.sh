#!/usr/bin/env bash

ROOT="/ROOT_DIR"

# activitynet dataset 
python preprocess/dump_frames.py --root_dir $ROOT --dataset "activitynet" --num_frames 16
python preprocess/dump_frames.py --root_dir $ROOT --dataset "activitynet" --num_frames 32
python preprocess/dump_frames.py --root_dir $ROOT --dataset "activitynet" --num_frames 64
python preprocess/dump_frames.py --root_dir $ROOT --dataset "activitynet" --num_frames 128
python preprocess/dump_frames.py --root_dir $ROOT --dataset "activitynet" --num_frames 256

# charades dataset 
python preprocess/dump_frames.py --root_dir $ROOT --dataset "activitynet" --num_frames 16
python preprocess/dump_frames.py --root_dir $ROOT --dataset "activitynet" --num_frames 32
python preprocess/dump_frames.py --root_dir $ROOT --dataset "charades" --num_frames 64
python preprocess/dump_frames.py --root_dir $ROOT --dataset "charades" --num_frames 128
python preprocess/dump_frames.py --root_dir $ROOT --dataset "charades" --num_frames 256
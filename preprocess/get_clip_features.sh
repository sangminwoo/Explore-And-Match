#!/usr/bin/env bash

ROOT="/ROOT_DIR"
CLIP_MODEL="ViT-B/32"

# activitynet dataset
python preprocess/clip_encoder.py --root $ROOT --dataset "activitynet" --clip_model $CLIP_MODEL --num_frames 64
python preprocess/clip_encoder.py --root $ROOT --dataset "activitynet" --clip_model $CLIP_MODEL --num_frames 128
python preprocess/clip_encoder.py --root $ROOT --dataset "activitynet" --clip_model $CLIP_MODEL --num_frames 256

# charades dataset
python preprocess/clip_encoder.py --root $ROOT --dataset "charades" --clip_model $CLIP_MODEL --num_frames 64
python preprocess/clip_encoder.py --root $ROOT --dataset "charades" --clip_model $CLIP_MODEL --num_frames 128
python preprocess/clip_encoder.py --root $ROOT --dataset "charades" --clip_model $CLIP_MODEL --num_frames 256
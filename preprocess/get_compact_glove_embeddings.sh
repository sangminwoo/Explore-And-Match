#!/usr/bin/env bash

ROOT="/ROOT_DIR"
GLOVE="/ROOT_DIR/glove/glove.6B.300d.txt"

# activitynet dataset 
python preprocess/glove.py --root_dir $ROOT --glove_dir $GLOVE --dataset "activitynet"

# charades dataset 
python preprocess/glove.py --root_dir $ROOT --glove_dir $GLOVE --dataset "charades"
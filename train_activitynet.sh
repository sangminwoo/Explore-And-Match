#!/bin/bash

root=/ROOT_DIR 
dataset=activitynet
data_type=features # features / raw
backbone=clip # clip / c3d_lstm
method=joint # joint / stepwise
end_epoch=200
bs=16
# num_videos
# train/val/test: 10009/4917/4885
# num_proposals
# train/val/test: 37421/17505/17031
lr=1e-4
lr_drop_step=50
num_input_frames=64
num_input_sentences=4
# num_sentences
# (train) avg: 3.739 / max: 27 / min: 2
# (val) avg: 3.560 / max: 25 / min: 2
# (test) avg: 3.486 / max:15 / min:3
enc_layers=4 # base: 2 / large: 4 / huge: 6
dec_layers=4 # base: 2 / large: 4 / huge: 6
num_proposals=$((10*$num_input_sentences)) # empirically gt*10 is good
set_cost_span=1
set_cost_giou=3
set_cost_query=2
pred_label=cos # att / sim / cos / pred
resume=/SAVE_DIR/best_model_\
${dataset}_${backbone}_${bs}b_${enc_layers}l_${num_input_frames}f_${num_proposals}q_\
${pred_label}_${set_cost_span}_${set_cost_giou}_${set_cost_query}.ckpt

PYTHONPATH=$PYTHONPATH:. python train.py \
--root ${root} \
--dataset ${dataset} \
--data_type ${data_type} \
--backbone ${backbone} \
--method ${method} \
--end_epoch ${end_epoch} \
--bs ${bs} \
--lr ${lr} \
--lr_drop_step ${lr_drop_step} \
--num_input_frames ${num_input_frames} \
--num_input_sentences ${num_input_sentences} \
--enc_layers ${enc_layers} \
--dec_layers ${dec_layers} \
--num_proposals ${num_proposals} \
--set_cost_span ${set_cost_span} \
--set_cost_giou ${set_cost_giou} \
--set_cost_query ${set_cost_query} \
--pred_label ${pred_label}
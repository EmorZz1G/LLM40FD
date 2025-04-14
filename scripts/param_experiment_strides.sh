#!/bin/bash

# for strides in "[1]" "[3]" "[5]" "[11]" "[1,3,5]" "[1,3,11]" "[1,5,11]" "[3,5,11]" "[1,3,5,11]"; do
# for strides in "[1]" "[3]" "[5]" "[11]" "[1,3,5]" "[1,3,11]" "[3,5,11]" "[1,3,5,11]"; do
# for strides in "[1]" "[3]" "[5]" "[11]" "[1,3,5]" "[1,3,11]" "[3,5,11]" "[1,3,5,11]"; do
for strides in "[40]"; do
    python main.py \
        --seed 2024 \
        --debug 0 \
        --saving_name "param_strides_${strides}" \
        --dataset1 "20k" \
        --dataset2 '["credit","shu"]' \
        --root_path "/share/home/202220143416/anomaly_data/fraud_detection/" \
        --enc_in 7 \
        --d_model 128 \
        --strides $strides \
        --llm_layers 6 \
        --llm_model "GPT2" \
        --llm_dim 768 \
        --device "cpu" \
        --num_workers 8 \
        --train_epochs 1 \
        --few_data_per 0.05 \
        --batch_size 128 \
        --learning_rate 0.0001 \
        --loss "OC_CE" \
        --pos_r 0.8 \
        --neg_r 2

    # python main.py \
    #     --seed 2024 \
    #     --debug 0 \
    #     --saving_name "param_strides_${strides}" \
    #     --dataset1 "credit" \
    #     --dataset2 '["credit","shu"]' \
    #     --root_path "/share/home/202220143416/anomaly_data/fraud_detection/" \
    #     --enc_in 7 \
    #     --d_model 128 \
    #     --strides $strides \
    #     --llm_layers 3 \
    #     --llm_model "GPT2" \
    #     --llm_dim 768 \
    #     --device "cuda" \
    #     --num_workers 8 \
    #     --train_epochs 1 \
    #     --few_data_per 1. \
    #     --batch_size 128 \
    #     --learning_rate 0.0001 \
    #     --loss "OC_CE" \
    #     --pos_r 0.8 \
    #     --neg_r 2
done
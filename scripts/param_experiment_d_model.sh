#!/bin/bash

for d_model in 16 32 64 128 256; do
    python main.py \
        --seed 2024 \
        --debug 0 \
        --saving_name "param_d_model_${d_model}" \
        --dataset1 "shu" \
        --dataset2 '["credit","shu"]' \
        --root_path "/share/home/202220143416/anomaly_data/fraud_detection/" \
        --enc_in 7 \
        --d_model $d_model \
        --strides [1,3,5,11] \
        --llm_layers 3 \
        --llm_model "GPT2" \
        --llm_dim 768 \
        --device "cuda" \
        --num_workers 8 \
        --train_epochs 1 \
        --few_data_per 1. \
        --batch_size 64 \
        --learning_rate 0.0001 \
        --loss "OC_CE" \
        --pos_r 0.8 \
        --neg_r 2

    python main.py \
        --seed 2024 \
        --debug 0 \
        --saving_name "param_d_model_${d_model}" \
        --dataset1 "credit" \
        --dataset2 '["credit","shu"]' \
        --root_path "/share/home/202220143416/anomaly_data/fraud_detection/" \
        --enc_in 7 \
        --d_model $d_model \
        --strides [1,3,5,11] \
        --llm_layers 3 \
        --llm_model "GPT2" \
        --llm_dim 768 \
        --device "cuda" \
        --num_workers 8 \
        --train_epochs 1 \
        --few_data_per 1. \
        --batch_size 64 \
        --learning_rate 0.0001 \
        --loss "OC_CE" \
        --pos_r 0.8 \
        --neg_r 2
done
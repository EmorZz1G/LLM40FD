#!/bin/bash
# --strides [1,3,5,7] \

# python main_dist.py \
#     --seed 2024 \
#     --debug 0 \
#     --zero_shot 1 \
#     --saving_name "TEST_DIST" \
#     --dataset1 "credit" \
#     --root_path "/share/home/202220143416/anomaly_data/fraud_detection/" \
#     --enc_in 7 \
#     --d_model 64 \
#     --strides [5] \
#     --llm_layers 3 \
#     --llm_model "GPT2" \
#     --llm_dim 768 \
#     --device "cuda" \
#     --num_workers 8 \
#     --train_epochs 1 \
#     --few_data_per 0.5 \
#     --batch_size 128 \
#     --learning_rate 0.0001 \
#     --loss "OC_CE" \
#     --pos_r 0.8 \
#     --neg_r 3

python main_dist.py \
    --seed 2024 \
    --debug 0 \
    --zero_shot 1 \
    --saving_name "TEST_DIST2" \
    --dataset1 "shu" \
    --root_path "/share/home/202220143416/anomaly_data/fraud_detection/" \
    --enc_in 7 \
    --d_model 128 \
    --strides [1,3,5,7] \
    --llm_layers 3 \
    --llm_model "GPT2" \
    --llm_dim 768 \
    --device "cuda" \
    --num_workers 8 \
    --train_epochs 10 \
    --few_data_per 1 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --loss "OC_CE" \
    --pos_r 0.5 \
    --neg_r 1.5

# python main_dist.py \
#     --seed 2024 \
#     --debug 0 \
#     --zero_shot 1 \
#     --saving_name "TEST_DIST" \
#     --dataset1 "credit" \
#     --root_path "/share/home/202220143416/anomaly_data/fraud_detection/" \
#     --enc_in 7 \
#     --d_model 512 \
#     --strides [1,3,5,7] \
#     --llm_layers 3 \
#     --llm_model "GPT2" \
#     --llm_dim 768 \
#     --device "cuda" \
#     --num_workers 8 \
#     --train_epochs 1 \
#     --few_data_per 0.5 \
#     --batch_size 128 \
#     --learning_rate 0.0001 \
#     --loss "OC" \
#     --pos_r 0.5 \
#     --neg_r -1



# python main_dist.py \
#     --seed 2024 \
#     --debug 0 \
#     --zero_shot 1 \
#     --saving_name "TEST_DIST" \
#     --dataset1 "shu" \
#     --root_path "/share/home/202220143416/anomaly_data/fraud_detection/" \
#     --enc_in 7 \
#     --d_model 512 \
#     --strides [1,3,5,7] \
#     --llm_layers 3 \
#     --llm_model "GPT2" \
#     --llm_dim 768 \
#     --device "cuda" \
#     --num_workers 8 \
#     --train_epochs 1 \
#     --few_data_per 1 \
#     --batch_size 128 \
#     --learning_rate 0.0001 \
#     --loss "OC" \
#     --pos_r 0.5 \
#     --neg_r -1
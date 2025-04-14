# for per in 0.05 0.1 0.3 0.5 1; do
# for per in 0.5 1; do
for per in 1; do
    for d1 in '20k' 'shu' 'credit' 'dh' 'jop';do
        python main.py \
        --seed 2024 \
        --debug 0 \
        --saving_name "zero_shot_${d1}_${per}" \
        --dataset1 "${d1}" \
        --root_path "/share/home/202220143416/anomaly_data/fraud_detection/" \
        --enc_in 7 \
        --d_model 128 \
        --strides [1,3,5,11] \
        --llm_layers 3 \
        --llm_model "GPT2" \
        --llm_dim 768 \
        --device "cuda" \
        --num_workers 8 \
        --train_epochs 1 \
        --few_data_per $per \
        --batch_size 64 \
        --learning_rate 0.0001 \
        --loss "OC_CE" \
        --pos_r 0.8 \
        --neg_r 2
    done 
done
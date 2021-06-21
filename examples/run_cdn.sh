#!/usr/bin/env bash
DATA_DIR="CBLUEDatasets"

TASK_NAME="cdn"
MODEL_TYPE="bert"
MODEL_DIR="data/model_data"
MODEL_NAME="chinese-bert-wwm-ext"
OUTPUT_DIR="data/output"
RESULT_OUTPUT_DIR="data/result_output"

MAX_LENGTH=64

RECALL_K=200
NUM_NEGATIVE_SAMPLES=5
DO_AUGMENT=6

echo "Start running"

if [ $# == 0 ]; then
    python baselines/run_cdn.py \
        --data_dir=${DATA_DIR} \
        --model_type=${MODEL_TYPE} \
        --model_dir=${MODEL_DIR} \
        --model_name=${MODEL_NAME} \
        --task_name=${TASK_NAME} \
        --output_dir=${OUTPUT_DIR} \
        --result_output_dir=${RESULT_OUTPUT_DIR} \
        --do_train \
        --recall_k=${RECALL_K} \
        --num_neg=${NUM_NEGATIVE_SAMPLES} \
        --do_aug=${DO_AUGMENT} \
        --max_length=${MAX_LENGTH} \
        --train_batch_size=32 \
        --eval_batch_size=256 \
        --learning_rate=4e-5 \
        --epochs=3 \
        --warmup_proportion=0.1 \
        --earlystop_patience=100 \
        --max_grad_norm=0.0 \
        --logging_steps=250 \
        --save_steps=250 \
        --seed=2021
elif [ $1 == "predict" ]; then
    python baselines/run_cdn.py \
        --data_dir=${DATA_DIR} \
        --model_type=${MODEL_TYPE} \
        --model_name=${MODEL_NAME} \
        --model_dir=${MODEL_DIR} \
        --task_name=${TASK_NAME} \
        --output_dir=${OUTPUT_DIR} \
        --recall_k=${RECALL_K} \
        --num_neg=${NUM_NEGATIVE_SAMPLES} \
        --do_aug=${DO_AUGMENT} \
        --result_output_dir=${RESULT_OUTPUT_DIR} \
        --do_predict \
        --max_length=${MAX_LENGTH} \
        --eval_batch_size=256 \
        --seed=2021
fi

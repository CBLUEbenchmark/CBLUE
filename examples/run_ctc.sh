#!/usr/bin/env bash
DATA_DIR="CBLUEDatasets"

TASK_NAME="ctc"
MODEL_TYPE="bert"
MODEL_DIR="data/model_data"
MODEL_NAME="chinese-roberta-large"
OUTPUT_DIR="data/output"
RESULT_OUTPUT_DIR="data/result_output"

MAX_LENGTH=50

echo "Start running"

if [ $# == 0 ]; then
    python baselines/run_classifier.py \
        --data_dir=${DATA_DIR} \
        --model_type=${MODEL_TYPE} \
        --model_dir=${MODEL_DIR} \
        --model_name=${MODEL_NAME} \
        --task_name=${TASK_NAME} \
        --output_dir=${OUTPUT_DIR} \
        --result_output_dir=${RESULT_OUTPUT_DIR} \
        --do_train \
        --max_length=${MAX_LENGTH} \
        --train_batch_size=24 \
        --eval_batch_size=64 \
        --learning_rate=2e-5 \
        --epochs=5 \
        --warmup_proportion=0.1 \
        --earlystop_patience=100 \
        --max_grad_norm=0.0 \
        --logging_steps=100 \
        --save_steps=100 \
        --seed=1000
elif [ $1 == "predict" ]; then
    python baselines/run_classifier.py \
        --data_dir=${DATA_DIR} \
        --model_type=${MODEL_TYPE} \
        --model_name=${MODEL_NAME} \
        --model_dir=${MODEL_DIR} \
        --task_name=${TASK_NAME} \
        --output_dir=${OUTPUT_DIR} \
        --result_output_dir=${RESULT_OUTPUT_DIR} \
        --do_predict \
        --max_length=${MAX_LENGTH} \
        --eval_batch_size=16 \
        --seed=2021
fi

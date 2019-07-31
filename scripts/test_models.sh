#!/bin/bash

# This is a convenience script for testing multiple models in a row on the same test dataset
# and writing their test results to the csv located at results_path. If you want to run this script on your own
# models you can simply edit the script below


ROOT=$(git rev-parse --show-toplevel)
RESULTS_PATH="./results.csv"

cd $ROOT
TEST_DIR="/home/chrisheinrich/data/dtu_7scenes_rgbd_scenes11_copy"

if [ $# -gt 0 ]; then
    TEST_DIR=$1
fi



python -m mvsnet.test --input_dir=$TEST_DIR --results_path=$RESULTS_PATH --wandb \
--model_dir gs://mvs-training-mlengine/dtu_pretrained_baseline_models/ --ckpt_step 100000

python -m mvsnet.test --input_dir=$TEST_DIR --results_path=$RESULTS_PATH --wandb \
--model_dir gs://mvs-training-mlengine/f_grad_loss/models/ --ckpt_step 140000


python -m mvsnet.test --input_dir=$TEST_DIR --results_path=$RESULTS_PATH --wandb \
--model_dir gs://mvs-training-mlengine/f_baseline/models/ --ckpt_step 140000






#!/bin/bash

# This is a convenience script for testing multiple models in a row on the same test dataset
# and writing their test results to the csv located at results_path. If you want to run this script on your own
# models you can simply edit the script below


ROOT=$(git rev-parse --show-toplevel)
RESULTS_PATH="./results.csv"

cd $ROOT
TEST_DIR="/Users/chrisheinrich/data/mvs_test"

python -m mvsnet.test --input_dir=$TEST_DIR --results_path=$RESULTS_PATH --wandb \
--model_dir=gs://mvs-training-mlengine/f_4gpu_alpha_0_25_epsilon_0_01_lr_0_0002_grad_unet/models/ \
--ckpt_step=340000 


python -m mvsnet.test --input_dir=$TEST_DIR --results_path=$RESULTS_PATH --wandb \
--model_dir=gs://mvs-training-mlengine/f_alpha_025_beta_0_continuation/models/ \
--ckpt_step=335000 


python -m mvsnet.test --input_dir=$TEST_DIR --results_path=$RESULTS_PATH --wandb \
--model_dir=gs://mvs-training-mlengine/f_alpha_025_beta_0/models/ \
--ckpt_step=140000


python -m mvsnet.test --input_dir=$TEST_DIR --results_path=$RESULTS_PATH --wandb \
--model_dir=gs://mvs-training-mlengine/f_baseline/models/ \
--ckpt_step=140000
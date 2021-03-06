#!/bin/bash

# This is a convenience script for qualitative testing on multiple models in a row on the same test dataset
# and writing the sketchfab links to the csv located at results_path. If you want to run this script on your own
# models you can simply edit the script below


ROOT=$(git rev-parse --show-toplevel)
cd $ROOT

RESULTS_PATH="./sketchfab_results.csv"
TEST_DIR="/home/chrisheinrich/data/atlas_hard"


python -m scripts.test_and_fuse --prob_threshold 0.1 --disp_threshold 0.2 --num_consistent 3 \
--test_folder_root $TEST_DIR --results_path $RESULTS_PATH \
--model_dir gs://mvs-training-mlengine/f_4gpu_alpha_0_25_epsilon_0_01_lr_0_0002_grad_unet/models/ --ckpt_step 340000


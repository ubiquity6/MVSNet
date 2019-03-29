

# Data generation tools for MVS deep learning 


## Introduction

This package includes data reading and data generation tools for preparing batches of data for multi view stereo. 

## Data read format

* This package reads data of the type exported by export_densify_frames.cpp in the U6 directory


## Data generation format

* ClusterGenerator generates batches of MVS data that can be read by MVSNet: https://github.com/YoYo000/MVSNet. That being said, it can easily be extended/adapted to deliver batches of data for virtually any MVS pipeline.

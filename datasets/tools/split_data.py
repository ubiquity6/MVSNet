import argparse
import json
from datasets.convert import utils
import random
import imageio
import numpy as np
import cv2
import shutil
import json
import os
import time


"""" 
This script splits a directory of mvs training sessions into a train / val / test datasets. 

"""


def split_data(data_dir):
    sessions = [f for f in os.listdir(args.data_dir) if not f.startswith(
        '.') if not f.endswith('.txt')]
    num_sessions = len(sessions)
    random.shuffle(sessions)
    n = 0
    for i, s in enumerate(sessions):


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help="Diretory where data is")
    parser.add_argument('--train', type=float, default=0.9,
                        help="Fraction of data to use for training")
    parser.add_argument('--val', type=str, default=0.05,
                        help="Fraction of data to use for validation")
    parser.add_argument('--test', type=str, default=0.05,
                        help="Diretory where data is")
    args = parser.parse_args()
    assert (args.train + args.val +
            args.test) == 1.0, 'Train, val and test fractions must add up to 1!'
    split_data(args)

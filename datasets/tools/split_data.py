import argparse
import random
import numpy as np
import shutil
import json
import os
import time


"""" 
This script splits a directory of mvs training sessions into a train / val / test datasets in separate directories. 

"""


def split_data(data_dir):
    sessions = [f for f in os.listdir(args.data_dir) if not f.startswith(
        '.') if not f.endswith('.txt')]
    num_sessions = len(sessions)
    num_train = int(np.floor(args.train * num_sessions))
    num_val = int(np.floor(args.val * num_sessions))
    num_test = int(np.floor(args.test * num_sessions))
    random.shuffle(sessions)
    train_sessions = sessions[:num_train]
    val_sessions = sessions[num_train:num_train + num_val]
    test_sessions = sessions[num_train + num_val:]
    print('{} total sessions'.format(num_sessions))
    print('{} train sessions'.format(num_train))
    print('{} val sessions'.format(num_val))
    print('{} test sessions'.format(num_test))
    test_dir = os.path.join(args.data_dir, 'test')
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    os.makedirs(test_dir)
    for i, s in enumerate(train_sessions):
        src = os.path.join(args.data_dir, s)
        dst = os.path.join(train_dir, s)
        shutil.move(src, dst)
    for i, s in enumerate(val_sessions):
        src = os.path.join(args.data_dir, s)
        dst = os.path.join(val_dir, s)
        shutil.move(src, dst)
    for i, s in enumerate(test_sessions):
        src = os.path.join(args.data_dir, s)
        dst = os.path.join(test_dir, s)
        shutil.move(src, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help="Diretory where data is")
    parser.add_argument('--train', type=float, default=0.9,
                        help="Fraction of data to use for training")
    parser.add_argument('--val', type=str, default=0.075,
                        help="Fraction of data to use for validation")
    parser.add_argument('--test', type=str, default=0.025,
                        help="Diretory where data is")
    args = parser.parse_args()
    assert (args.train + args.val +
            args.test) == 1.0, 'Train, val and test fractions must add up to 1!'
    split_data(args)

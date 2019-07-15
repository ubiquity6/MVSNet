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
Fixes DTU data

- Resizes depth image to same size as images
- Updates focal lengths to correct values

"""


def fix_depths(data_dir):
    sessions = [f for f in os.listdir(data_dir) if not f.startswith(
        '.') if not f.endswith('.txt')]
    num_sessions = len(sessions)
    random.shuffle(sessions)
    n = 0
    for i, s in enumerate(sessions):
        if 'dtu_scan' in s:
            print('Fixing session {} '.format(s))
            sdir = os.path.join(data_dir, s)
            depths_dir = os.path.join(sdir, 'depths')
            depths = os.listdir(depths_dir)
            contains_uint8 = False
            for j in range(len(depths)):
                depth_path = os.path.join(depths_dir, depths[j])
                img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                res = cv2.resize(img, dsize=(640, 512),
                                 interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(depth_path, res)
            cameras_dir = os.path.join(sdir, 'cameras')
            cameras = os.listdir(cameras_dir)
            for c in cameras:
                focal_rescale = 1.171875
                camera_path = os.path.join(cameras_dir, c)
                camera_data = {}
                with open(camera_path, mode='r') as f:
                    camera_data = json.load(f)
                camera_data["intrinsics"]["fx"] = camera_data["intrinsics"]["fx"] * focal_rescale
                camera_data["intrinsics"]["fy"] = camera_data["intrinsics"]["fy"] * focal_rescale
                with open(camera_path, 'w') as g:
                    json.dump(camera_data, g)
            if n % 20 == 0:
                print('Fixed {} sessions'.format(n))
            n += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help="Diretory where data is")
    args = parser.parse_args()
    fix_depths(args.data_dir)

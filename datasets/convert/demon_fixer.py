import argparse
import json
from datasets.convert import utils
import random
import imageio
import numpy as np
import shutil
import json
import os


"""" 
Fixes DeMoN data

- Removes clusters with depth saved as uint8
- Ensures min depth and max depth are calculated properly

"""


def fix_depths(data_dir):
    sessions = [f for f in os.listdir(data_dir) if not f.startswith(
        '.') if not f.endswith('.txt')]
    num_sessions = len(sessions)
    random.shuffle(sessions)
    for i, s in enumerate(sessions):
        try:
            sdir = os.path.join(data_dir, s)
            depths_dir = os.path.join(sdir, 'depths')
            depths = os.listdir(depths_dir)
            dmin = 400
            dmax = 10000
            contains_uint8 = False
            for j in range(len(depths)):
                depth_path = os.path.join(depths_dir, depths[j])
                data = imageio.imread(depth_path)
                d_max = data[data != 65535].max()
                d_min = data[data != 0].min()
                if d_max > dmax:
                    dmax = d_max
                if d_min < dmin:
                    dmin = d_min
                if data.dtype == np.uint8:
                    contains_uint8 = True

            if contains_uint8:
                print(
                    "Found a depth image with uint8 at cluster {}. Deleting cluster!".format(sdir))
                shutil.rmtree(sdir)
            else:
                covis_path = os.path.join(sdir, 'covisibility.json')
                with open(covis_path, 'r') as f:
                    covis = json.load(f)
                for k in list(covis.keys()):
                    covis[k]['min_depth'] = int(dmin)
                    covis[k]['max_depth'] = int(dmax)
                with open(covis_path, 'w') as fw:
                    json.dump(covis, fw)

                if i % 25 == 0:
                    print('Fixed {} of {} sessions'.format(i, num_sessions))
        except Exception as e:
            print(
                'Failed to fix session {} with exception {}. Removing sessions'.format(s, e))
            try:
                sdir = os.path.join(data_dir, s)
                shutil.rmtree(sdir)
            except Exception as e:
                print('Failed to remove session {}'.format(s))
       # print(covis)
       # print('Min depth for session {} is {}'.format(s, dmin))
       # print('Max depth for session {} is {}'.format(s, dmax))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help="Diretory where data is")
    args = parser.parse_args()
    fix_depths(args.data_dir)

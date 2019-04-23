import argparse
import json
from datasets.convert import utils
import random
import os


"""" 
Convert DeMoN data to MVS Training

DeMoN data is expected to have been downloaded and prepared using DPSNet's 
downloader for DeMoN -- see https://github.com/sunghoonim/DPSNet
After this is downlaoded and prepared, run this script on that directory of data
to convert to the mvs_training format which is documented in export_densify_frames.cpp

"""


def convert_demon(data_dir):
    sessions = [f for f in os.listdir(data_dir) if not f.startswith(
        '.') if not f.endswith('.txt')]
    num_sessions = len(sessions)
    random.shuffle(sessions)
    for i, s in enumerate(sessions):
        sdir = os.path.join(data_dir, s)
        num_depths, mind, maxd = utils.depths_from_demon(sdir)
        num_cams = utils.cameras_from_demon(sdir)
        num_images = utils.images_from_demon(sdir)
        if num_depths != num_cams or num_cams != num_images:
            print('WARN not all depths/cams/images present!')
        else:
            utils.covisibility_from_demon(sdir, mind, maxd)
            print('Processed session {}/{} at {}'.format(i, num_sessions, s))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help="Diretory where data is")
    args = parser.parse_args()
    convert_demon(args.data_dir)

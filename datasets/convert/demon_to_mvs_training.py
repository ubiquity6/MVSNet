import argparse
import json
import utils
import os


def convert_demon(data_dir):
    sessions = utils.list_no_hidden(data_dir)
    num_sessions = len(sessions)
    for i, s in enumerate(sessions):
        print('Processing session {}/{} at {}'.format(i, num_sessions, s))
        sdir = os.path.join(data_dir, s)
        num_depths, mind, maxd = utils.depths_from_demon(sdir)
        num_cams = utils.cameras_from_demon(sdir)
        num_images = utils.images_from_demon(sdir)
        if num_depths != num_cams or num_cams != num_images:
            print('WARN not all depths/cams/images present!')
        utils.covisibility_from_demon(sdir, mind, maxd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help="Diretory where data is")
    args = parser.parse_args()
    convert_demon(args.data_dir)

import os
import subprocess
import argparse
import utils as ut
import time


dense_folders = [
    '../data/7scenes/test/office_9_mvs_training/',
    '../data/7scenes/test/fire_4_mvs_training/',
    '../data/7scenes/test/redkitchen_14_mvs_training/',
    '../data/7scenes/test/stairs_4_mvs_training/',
    '../data/7scenes/test/chess_5_mvs_training/',
    '../data/7scenes/test/heads_1_mvs_training/',
    '../data/7scenes/test/umpkin_7_mvs_training/',
]

all_urls = []

def main(args):
    start_time = time.time()
    ply_folder = os.path.join(args.ply_folder,start_time)
    for d in os.listdir(args.test_folder_root):
        dense_folder = os.path.join(args.test_folder_root, d)
        if args.no_test is not True:
            ut.test(dense_folder)
        ut.clear_old_points(dense_folder)
        ut.fuse(dense_folder, args.fusibile_path, args.prob_threshold, args.disp_threshold, args.num_consistent)
        ply_paths = ut.get_fusion_plys(dense_folder)
        urls = ut.handle_plys(ply_paths, dense_folder, ply_folder)
        all_urls.append(urls)
    print('Models uploaded to:',all_urls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_folder_root', type=str, default='../data/7scenes/test /')
    parser.add_argument('--fusibile_path', type=str,
                        default='/home/chrisheinrich/fusibile/fusibile')
    parser.add_argument('--prob_threshold', type=float, default='0.6')
    parser.add_argument('--ply_folder', type=str,
                        default='/home/chrisheinrich/fused-point-clouds')
    parser.add_argument('--disp_threshold', type=float, default='0.15')
    parser.add_argument('--num_consistent', type=float, default='3')
    parser.add_argument('--no_test', action='store_true', help='Will not run testing, but only postprocessing, if flag is set')
    args = parser.parse_args()
    main(args)


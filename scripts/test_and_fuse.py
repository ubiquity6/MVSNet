import os
import subprocess
import argparse
import utils as ut


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
    if args.no_test is not True:
        ut.test(args.dense_folder)
    ut.fuse(args.dense_folder, args.fusibile_path, args.prob_threshold, args.disp_threshold, args.num_consistent)
    ply_paths = ut.get_fusion_plys(args.dense_folder)
    urls = ut.handle_plys(ply_paths, args.dense_folder, args.ply_folder)
    all_urls.append(urls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dense_folder', type=str, default='')
    parser.add_argument('--fusibile_path', type=str,
                        default='/home/chrisheinrich/fusibile/fusibile')
    parser.add_argument('--prob_threshold', type=float, default='0.1')
    parser.add_argument('--ply_folder', type=str,
                        default='/home/chrisheinrich/fused-point-clouds')
    parser.add_argument('--disp_threshold', type=float, default='0.1')
    parser.add_argument('--num_consistent', type=float, default='2')
    parser.add_argument('--no_test', action='store_true', help='Will not run testing, but only postprocessing, if flag is set')
    args = parser.parse_args()
    for f in dense_folders:
        args.dense_folder = f
        args.no_test = True
        print('Args:', args)
        main(args)
    print('Models uploaded to URLS:', all_urls)


import os
import subprocess
import argparse
import utils as ut
import time

"""
A simple script for running prediction on a multiple sessions, fusing the resulting point clouds,
and then uploading the results to sketchfab, as well as copying them to a more convenient location
on the file system.
"""


def write_results(args, urls):
    try:
        with open(args.results_path, 'a+') as f:
            new_line = '{}, {}, {}, {}, {}, {} \n'.format(
                args.model_dir, args.ckpt_step, urls, args.prob_threshold, args.disp_threshold, args.num_consistent)
            f.write(new_line)
    except Exception as e:
        logger.error('Failed to write results with exception {}'.format(e))
        pass  # While it is too bad if results fail to write, we don't want to stop the process over it


def test_and_fuse(args, dense_folder, ply_folder):
    if args.no_test is not True:
        ut.test(dense_folder, args.ckpt_step, args.model_dir)
    ut.clear_old_points(dense_folder)
    ut.fuse(dense_folder, args.fusibile_path, args.depth_folder_name, args.prob_threshold,
            args.disp_threshold, args.num_consistent)
    ply_paths = ut.get_fusion_plys(dense_folder)
    urls = ut.handle_plys(ply_paths, dense_folder, ply_folder, args)
    print('Sketchfab urls {}'.format(urls))
    write_results(args, urls)
    return urls


def main(args):
    all_urls = []
    start_time = time.time()
    dir_name = '{}_prob_{}_disp_{}_consis_{}'.format(start_time,
                                                     args.prob_threshold, args.disp_threshold, args.num_consistent)
    ply_folder = os.path.join(args.ply_folder, dir_name)
    print('Final PLY files will be written to {}'.format(ply_folder))
    os.mkdir(ply_folder)

    # If test_data_root is a session dir we test on that, otherwise we test on subdirs
    if os.path.isfile(os.path.join(args.test_folder_root, 'covisibility.json')):
        urls = test_and_fuse(args, args.test_folder_root, ply_folder)
        all_urls.append(urls)
    else:
        for d in os.listdir(args.test_folder_root):
            dense_folder = os.path.join(args.test_folder_root, d)
            try:
                urls = test_and_fuse(args, dense_folder, ply_folder)
                all_urls.append(urls)
            except Exception as e:
                print('Failed to test and fuse on dense folder {}'.format(dense_folder))
    print('Models uploaded to:', all_urls)
    write_results(args, all_urls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_step', type=str,
                        help="The ckpt_step of saved model -- see test.py")
    parser.add_argument('--model_dir', type=str,
                        help="The directory of saved model -- see test.py")
    parser.add_argument('--test_folder_root', type=str,
                        default='../data/atlas', help="The directory where the sessions to be tested are located")
    parser.add_argument('--fusibile_path', type=str,
                        default='/home/chrisheinrich1/fusibile/fusibile', help="The path to the compiled fusibile executable")
    parser.add_argument('--prob_threshold', type=float, default='0.8')
    parser.add_argument('--ply_folder', type=str,
                        default='/home/chrisheinrich1/fused-point-clouds', help="The root directory for storing the saved point cloud output")
    parser.add_argument('--disp_threshold', type=float, default='0.25')
    parser.add_argument('--num_consistent', type=float, default='3')
    parser.add_argument('--no_test', action='store_true',
                        help='Will not run testing, but only postprocessing, if flag is set')
    parser.add_argument('--test_only', action='store_true',
                        help='Will only run testing, and no fusing or uploading of point clouds.')
    parser.add_argument('--results_path', type=str,
                        default='./sketchfab_links.csv', help="The path to where to write teh sketchfab results")
    parser.add_argument('--depth_folder_name', type=str,
                        default='depths_mvsnet')
    args = parser.parse_args()
    main(args)

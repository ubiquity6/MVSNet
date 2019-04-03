import os
import subprocess
import argparse
import sketchfab
import shutil
from shutil import copyfile


def run(args):
    subprocess.call(args)


def test(dense_folder):
     args = ["python", "-m", "mvsnet.test", "--dense_folder",dense_folder]
     run(args)


def fuse(dense_folder, fusibile_path, prob_threshold = '0.1', disp_threshold='0.1',num_consistent='2'):
     args = ["python", "-m", "mvsnet.depthfusion", "--dense_folder", dense_folder, "--prob_threshold", \
             str(prob_threshold), "--disp_threshold", str(disp_threshold), "--num_consistent", str(num_consistent), '--fusibile_exe_path',fusibile_path]
     run(args)


def clear_old_points(dense_folder):
    points_dir = os.path.join(dense_folder, 'points_mvsnet')
    shutil.rmtree(points_dir)


def get_fusion_plys(dense_folder):
    ply_paths = []
    points_dir = os.path.join(dense_folder, 'points_mvsnet')
    sub_dirs = os.listdir(points_dir)
    for d in sub_dirs:
        if 'consistencyCheck' in d:
            ply_path = os.path.join(points_dir,d,'final3d_model.ply')
            ply_paths.append(ply_path)
    return ply_paths


def handle_plys(ply_paths, dense_folder, ply_folder):
    try:
        name = dense_folder.split('/')[-1]
        if name == '':
            name = dense_folder.split('/')[-2]
    except:
        print('Failed to get name from dense folder')
        name = 'model'
    urls = []
    for p in ply_paths:
        url = sketchfab.upload(p,name=name)
        urls.append(url)
        file_name = name + '.ply'
        dst = os.path.join(ply_folder, file_name)
        copyfile(p,dst)
    return urls




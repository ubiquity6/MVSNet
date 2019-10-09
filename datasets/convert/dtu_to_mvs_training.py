from mvsnet import preprocess as pp
import imageio
import argparse
import json
import utils
import os

"""
Converts DTU depth data from the format that is consumed by the original MVSNet
to the mvs-training format that is created by export_densify_frames
"""


    # 3 sets
training_set = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
                45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
                74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                121, 122, 123, 124, 125, 126, 127, 128]
validation_set = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]
evaluation_set = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 
                    110, 114, 118]


def convert_dtu(dtu_dir, output_dir):
    camera_base = os.path.join(dtu_dir, 'Cameras')
    camera_dir = os.path.join(dtu_dir, 'Cameras')
    depths_base = os.path.join(dtu_dir, 'Depths')
    images_base = os.path.join(dtu_dir, 'Rectified')
    pair_path = os.path.join(camera_base, 'pair.txt')
    num_scans = len((utils.list_no_hidden(images_base)))
    print("Number of scans = ", num_scans)
    for index, scan in enumerate(sorted(utils.list_no_hidden(images_base))):
        if index > 43:
            print("Processing scan", index)
            # For each dtu scan session there are 7 different lighting settings
            for l in range(7):
                session_dir = os.path.join(
                    output_dir, 'dtu_scan_{}_lighting_{}'.format(index, l))
                os.makedirs(session_dir)
                session_images = os.path.join(session_dir, 'images')
                session_depths = os.path.join(session_dir, 'depths')
                session_cams = os.path.join(session_dir, 'cameras')
                os.makedirs(session_images)
                os.makedirs(session_depths)
                os.makedirs(session_cams)
                covis_path = os.path.join(session_dir, 'covisibility.json')
                depths_dir = os.path.join(depths_base, scan)
                images_dir = os.path.join(images_base, scan)
                utils.pair_to_covisibility(pair_path, covis_path)
                for i in range(49):
                    txt_path = os.path.join(camera_dir, utils.cam_name(i))
                    json_path = os.path.join(session_cams, '{}.json'.format(i))
                    # cams need to be rescaled due to image resizing
                    rescale = 512.0 / 1200.0
                    utils.cam_to_json(txt_path, json_path,
                                      scale_factor=rescale)
                for j in range(49):
                    png_path = os.path.join(session_depths, '{}.png'.format(j))
                    pfm_path = os.path.join(depths_dir, utils.depth_name(j))
                    utils.depth_pfm_to_png(pfm_path, png_path)
                    image_path = os.path.join(
                        images_dir, utils.image_name(j, l))
                    final_image_path = os.path.join(
                        session_images, '{}.jpg'.format(j))
                    img = imageio.imread(image_path)
                    imageio.imwrite(final_image_path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dtu_dir', type=str,
                        help="Diretory where dtu data is")
    parser.add_argument('output_dir', type=str,
                        help="Directory to output the converted data")
    args = parser.parse_args()
    convert_dtu(args.dtu_dir, args.output_dir)

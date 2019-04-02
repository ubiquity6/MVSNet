#!/usr/bin/env python

from __future__ import print_function

import os
import errno
import time
import math
import re
import sys
import scipy
import imageio
import cv2
import numpy as np
import json
from random import Random
import logging


"""

Helper package for reading and writing MVS depth map training data

"""

"""
Copyright 2019, Chris Heinrich, Ubiquity6.
"""


def set_log_level(logger):
    if 'LOG_LEVEL' in os.environ:
        level = os.environ['LOG_LEVEL'].upper()
        exec('logger.setLevel(logging.{})'.format(level))
    else:
        logger.setLevel(logging.INFO)


# Set up logger
logger = logging.getLogger('data-generator-utils')
set_log_level(logger)


def center_image(img):
    """ normalize image input """
    img = img.astype(np.float32)
    var = np.var(img, axis=(0, 1), keepdims=True)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)


def center_images(images):
    for i in range(len(images)):
        images[i] = center_image(images[i])
    return images


def copy_and_center_images(ims):
    """ Returns a copy of the images, centered"""
    images = []
    for i in range(len(ims)):
        images.append(center_image(ims[i]))
    return images


def center_image_cluster(img):
    """ normalize image input. Same as above, except there are assumed
    to be a block of images stacked along the 0 axis"""
    img = img.astype(np.float32)
    std = np.std(img)
    mean = np.mean(img)
    return (img - mean) / (std + 0.00000001)


def scale_camera(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    # focal:
    new_cam[1][0][0] = cam[1][0][0] * scale
    new_cam[1][1][1] = cam[1][1][1] * scale
    # principle point:
    new_cam[1][0][2] = cam[1][0][2] * scale
    new_cam[1][1][2] = cam[1][1][2] * scale
    return new_cam


def scale_mvs_camera(cams, scale=1):
    """ resize input in order to produce sampled depth map """
    for view in range(len(cams)):
        cams[view] = scale_camera(cams[view], scale=scale)
    return cams


def scale_image(image, scale=1, interpolation='linear'):
    """ resize image using cv2 """
    if interpolation == 'linear':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'nearest':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def scale_and_reshape_depth(depth, output_scale):
    # Scale depth image to output_scale * image_scale
    depth = scale_image(
        depth, scale=output_scale, interpolation='nearest')
    # Increase rank of depth array and set shape[2] = 1
    depth_shape = (
        depth.shape[0], depth.shape[1], 1)
    depth = np.reshape(depth, depth_shape)
    return depth


def scale_mvs_input(images, cams, depth_image=None, scale=1):
    """ resize input to fit into the memory """
    for view in range(len(images)):
        images[view] = scale_image(images[view], scale=scale)
        cams[view] = scale_camera(cams[view], scale=scale)

    if depth_image is None:
        return images, cams
    else:
        depth_image = scale_image(
            depth_image, scale=scale, interpolation='nearest')
        return images, cams, depth_image


def crop_mvs_input(images, cams, max_w, max_h, base_image_size, depth_image=None):
    """ resize images and cameras to fit the network (so both dimensions are divisible by base_image_size ) """

    # crop images and cameras
    for view in range(len(images)):
        h, w = images[view].shape[0:2]
        new_h = h
        new_w = w
        if new_h > max_h:
            new_h = max_h
        else:
            new_h = int(math.ceil(h / base_image_size)
                        * base_image_size)
        if new_w > max_w:
            new_w = max_w
        else:
            new_w = int(math.ceil(w / base_image_size)
                        * base_image_size)
        start_h = int(math.ceil((h - new_h) / 2))
        start_w = int(math.ceil((w - new_w) / 2))
        finish_h = start_h + new_h
        finish_w = start_w + new_w
        images[view] = images[view][start_h:finish_h, start_w:finish_w]
        # Shift the principal point
        cams[view][1][0][2] = cams[view][1][0][2] - start_w
        cams[view][1][1][2] = cams[view][1][1][2] - start_h

    # crop depth image
    if not depth_image is None:
        depth_image = depth_image[start_h:finish_h, start_w:finish_w]
        return images, cams, depth_image
    else:
        return images, cams


def mask_depth_image(depth_image, min_depth, max_depth):
    """ mask out-of-range pixel to zero """
    ret, depth_image = cv2.threshold(
        depth_image, min_depth, 100000, cv2.THRESH_TOZERO)
    ret, depth_image = cv2.threshold(
        depth_image, max_depth, 100000, cv2.THRESH_TOZERO_INV)
    depth_image = np.expand_dims(depth_image, 2)
    return depth_image


def flip_cams(cams, depth_num):
    """ Modifies cams to be compatible with MVSNet GRU regularization"""
    cams[0][1, 3, 0] = cams[0][1, 3, 0] + \
        (depth_num - 1) * cams[0][1, 3, 1]
    cams[0][1, 3, 1] = -cams[0][1, 3, 1]
    return cams


def write_cam(file, cam):
    # f = open(file, "w")
    f = file_io.FileIO(file, "w")

    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + '\n')

    f.close()


def write_depth_map(file_path, image):
    # convert to int and clip to range of [0, 2^16 -1]
    image = np.clip(image, 0, 65535).astype(np.uint16)
    imageio.imsave(file_path, image)
    file_path_scaled = file_path.replace('.png', '_scaled.png')
    # Rescales the image so max distance is 6.5 meters, making contrast more visible
    # This is purely for visualization purposes
    depth_scale = 20
    image_scaled = np.clip(image*depth_scale, 0, 65535).astype(np.uint16)
    imageio.imsave(file_path_scaled, image_scaled)


def write_confidence_map(file_path, image):
    # we convert probabilities in range [0,1] to ints in range [0, 2^16-1]
    scale_factor = 65535
    image *= scale_factor
    image = np.clip(image, 0, 65535).astype(np.uint16)
    imageio.imsave(file_path, image)


def write_image(file_path, image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    imageio.imsave(file_path, image.astype(np.uint8))


def mkdir_p(dirname):
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

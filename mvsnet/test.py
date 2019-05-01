#!/usr/bin/env python

from __future__ import print_function
from loss import *
from model import *
from preprocess import *
from cnn_wrapper.common import Notify
from mvs_data_generation.cluster_generator import ClusterGenerator
"""
Copyright 2019, Yao Yao, HKUST.
Test script.
"""


import os
import time
import sys
import math
import argparse
import numpy as np
import imageio

import cv2
import tensorflow as tf

sys.path.append("../")

# dataset parameters
tf.app.flags.DEFINE_string('dense_folder', None,
                           """Root path to dense folder.""")
tf.app.flags.DEFINE_string('model_dir',
                           None,
                           """Path to restore the model.""")
tf.app.flags.DEFINE_integer('ckpt_step', None,
                            """ckpt  step.""")
# input parameters
tf.app.flags.DEFINE_integer('view_num', 6,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 200,
                            """Maximum depth step when testing.""")
tf.app.flags.DEFINE_integer('max_w', 640,
                            """Maximum image width when testing.""")
tf.app.flags.DEFINE_integer('max_h', 480,
                            """Maximum image height when testing.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25,
                          """Downsample scale for building cost volume (W and H).""")
tf.app.flags.DEFINE_float('interval_scale', 1.0,
                          """Downsample scale for building cost volume (D).""")
tf.app.flags.DEFINE_float('base_image_size', 8,
                          """Base image size""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Testing batch size.""")
tf.app.flags.DEFINE_bool('adaptive_scaling', True,
                         """Let image size to fit the network, including 'scaling', 'cropping'""")

# network architecture
tf.app.flags.DEFINE_string('regularization', '3DCNNs',
                           """Regularization method, including '3DCNNs' and 'GRU'""")
tf.app.flags.DEFINE_boolean('refinement', False,
                            """Whether to apply depth map refinement for MVSNet""")
tf.app.flags.DEFINE_bool('inverse_depth', True,
                         """Whether to apply inverse depth for R-MVSNet""")

FLAGS = tf.app.flags.FLAGS


def mvsnet_pipeline(test_folder, mvs_list=None):
    """ mvsnet in altizure pipeline """

    # create output folder
    output_folder = os.path.join(test_folder, 'depths_mvsnet')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # testing set
    data_gen = ClusterGenerator(test_folder, FLAGS.view_num, FLAGS.max_w, FLAGS.max_h,
                                FLAGS.max_d, FLAGS.interval_scale, FLAGS.base_image_size, mode='test')
    mvs_generator = iter(data_gen)
    sample_size = len(data_gen.train_clusters)

    generator_data_type = (tf.float32, tf.float32, tf.float32, tf.int32)
    mvs_set = tf.data.Dataset.from_generator(
        lambda: mvs_generator, generator_data_type)
    mvs_set = mvs_set.batch(FLAGS.batch_size)
    mvs_set = mvs_set.prefetch(buffer_size=1)

    # data from dataset via iterator
    mvs_iterator = mvs_set.make_initializable_iterator()
    scaled_images, centered_images, scaled_cams, image_index = mvs_iterator.get_next()

    # set shapes
    scaled_images.set_shape(tf.TensorShape(
        [None, FLAGS.view_num, None, None, 3]))
    centered_images.set_shape(tf.TensorShape(
        [None, FLAGS.view_num, None, None, 3]))
    scaled_cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
    depth_start = tf.reshape(
        tf.slice(scaled_cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    depth_interval = tf.reshape(
        tf.slice(scaled_cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    depth_num = tf.cast(
        tf.reshape(tf.slice(scaled_cams, [0, 0, 1, 3, 2], [1, 1, 1, 1, 1]), []), 'int32')

    # deal with inverse depth
    if FLAGS.regularization == '3DCNNs' and FLAGS.inverse_depth:
        depth_end = tf.reshape(
            tf.slice(scaled_cams, [0, 0, 1, 3, 3], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    else:
        depth_end = depth_start + \
            (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # depth map inference using 3DCNNs
    if FLAGS.regularization == '3DCNNs':
        init_depth_map, prob_map = inference_mem(
            centered_images, scaled_cams, FLAGS.max_d, depth_start, depth_interval)

        if FLAGS.refinement:
            ref_image = tf.squeeze(
                tf.slice(centered_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
            refined_depth_map = depth_refine(
                init_depth_map, ref_image, FLAGS.max_d, depth_start, depth_interval, True)

    # depth map inference using GRU
    elif FLAGS.regularization == 'GRU':
        init_depth_map, prob_map = inference_winner_take_all(centered_images, scaled_cams,
                                                             depth_num, depth_start, depth_end, reg_type='GRU', inverse_depth=FLAGS.inverse_depth)

    # init option
    init_op = tf.global_variables_initializer()
    var_init_op = tf.local_variables_initializer()

    # GPU grows incrementally
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 1

    with tf.Session(config=config) as sess:

        # initialization
        sess.run(var_init_op)
        sess.run(init_op)
        total_step = 0

        # load model
        if FLAGS.model_dir is not None:
            pretrained_model_ckpt_path = os.path.join(
                FLAGS.model_dir, FLAGS.regularization, 'model.ckpt')
            restorer = tf.train.Saver(tf.global_variables())
            restorer.restore(
                sess, '-'.join([pretrained_model_ckpt_path, str(FLAGS.ckpt_step)]))
            print(Notify.INFO, 'Pre-trained model restored from %s' %
                  ('-'.join([pretrained_model_ckpt_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
            total_step = FLAGS.ckpt_step

        # run inference for each reference view
        sess.run(mvs_iterator.initializer)
        for step in range(sample_size):

            start_time = time.time()
            try:
                out_init_depth_map, out_prob_map, out_images, out_cams, out_index = sess.run(
                    [init_depth_map, prob_map, scaled_images, scaled_cams, image_index])
            except tf.errors.OutOfRangeError:
                print("all dense finished")  # ==> "End of dataset"
                break
            duration = time.time() - start_time
            print(Notify.INFO, 'depth inference %d finished. (%.3f sec/step)' % (step, duration),
                  Notify.ENDC)

            # squeeze output
            out_init_depth_image = np.squeeze(out_init_depth_map)
            out_prob_map = np.squeeze(out_prob_map)
            out_ref_image = np.squeeze(out_images)
            out_ref_image = np.squeeze(out_ref_image[0, :, :, :])
            out_ref_cam = np.squeeze(out_cams)
            out_ref_cam = np.squeeze(out_ref_cam[0, :, :, :])
            out_index = np.squeeze(out_index)

            # paths
            init_depth_map_path = os.path.join(
                output_folder, '{}_init.pfm'.format(out_index))
            prob_map_path = os.path.join(
                output_folder, '{}_prob.pfm'.format(out_index))
            out_ref_image_path = os.path.join(
                output_folder, '{}.jpg'.format(out_index))
            out_ref_cam_path = os.path.join(
                output_folder, '{}.txt'.format(out_index))
            # png outputs
            prob_png = os.path.join(
                output_folder, '{}_prob.png'.format(out_index))
            depth_png = os.path.join(
                output_folder, '{}_depth.png'.format(out_index))

            # save output
            write_pfm(init_depth_map_path, out_init_depth_image)
            write_pfm(prob_map_path, out_prob_map)

            # for png outputs
            write_depth_map(depth_png, out_init_depth_image)
            write_confidence_map(prob_png, out_prob_map)

            out_ref_image = cv2.cvtColor(out_ref_image, cv2.COLOR_RGB2BGR)
            image_file = file_io.FileIO(out_ref_image_path, mode='w')
            scipy.misc.imsave(image_file, out_ref_image)
            write_cam(out_ref_cam_path, out_ref_cam)
            total_step += 1


def main(_):  # pylint: disable=unused-argument
    """ program entrance """
    # Acceptable input for the dense_folder is a single test folder, or a folder containing multiple
    # test folders. We check to see which one it is
    if os.path.isfile(os.path.join(FLAGS.dense_folder, 'covisibility.json')):
        mvsnet_pipeline(FLAGS.dense_folder)
    else:
        folders = os.listdir(FLAGS.dense_folder)
        for f in folders:
            mvsnet_pipeline(os.path.join(FLAGS.dense_folder, f))


if __name__ == '__main__':
    tf.app.run()

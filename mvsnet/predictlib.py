#!/usr/bin/env python

from __future__ import print_function
import os
import time
import sys
import math
import argparse
import numpy as np
import imageio
import cv2
# import wandb
import tensorflow as tf
from mvsnet.loss import *
from mvsnet.model import inference_mem, depth_refine, inference_winner_take_all
from mvsnet.preprocess import *
from mvsnet.cnn_wrapper.common import Notify
from mvsnet.mvs_data_generation.cluster_generator import ClusterGenerator
import mvsnet.utils as mu
from mvsnet.mvs_data_generation.utils import scale_image
logger = mu.setup_logger('predictlib')
sys.path.append("../")
tf.app.flags.DEFINE_bool('wandb', False,
                         """Whether or not to log inference results to wandb""")

FLAGS = tf.app.flags.FLAGS

""" 
A small library of helper functions for performing prediction with mvsnet 
"""


def setup_data_iterator(input_dir):
    "Configures the data generator that is used to feed batches of data for inference"
    mode = 'test' if FLAGS.benchmark else 'inference'
    data_gen = ClusterGenerator(input_dir, FLAGS.view_num, FLAGS.width, FLAGS.height, FLAGS.max_d, FLAGS.interval_scale,
                                FLAGS.base_image_size, mode=mode, output_scale=FLAGS.sample_scale, max_clusters_per_session=FLAGS.max_clusters_per_session)
    mvs_generator = iter(data_gen)
    sample_size = len(data_gen.clusters)

    if FLAGS.benchmark:
        generator_data_type = (tf.float32, tf.float32,
                               tf.float32, tf.float32, tf.float32, tf.int32, tf.string)
    else:
        generator_data_type = (tf.float32, tf.float32,
                               tf.float32, tf.float32, tf.int32)

    mvs_set = tf.data.Dataset.from_generator(
        lambda: mvs_generator, generator_data_type)
    mvs_set = mvs_set.batch(FLAGS.batch_size)
    mvs_set = mvs_set.prefetch(buffer_size=1)

    # data from dataset via iterator
    mvs_iterator = mvs_set.make_initializable_iterator()
    return mvs_iterator, sample_size


def setup_output_dir(input_dir, output_dir):
    "Creates output dir for saving mvsnet output"
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'depths_mvsnet')
    mu.mkdir_p(output_dir)
    logger.info('Running inference on {} and writing output to {}'.format(
        input_dir, output_dir))
    return output_dir


def load_model(sess):
    """Load trained model for inference """
    if FLAGS.model_dir is not None:
        pretrained_model_ckpt_path = os.path.join(
            FLAGS.model_dir, FLAGS.regularization, 'model.ckpt')
        restorer = tf.train.Saver(tf.global_variables())
        restorer.restore(
            sess, '-'.join([pretrained_model_ckpt_path, str(FLAGS.ckpt_step)]))
        print(Notify.INFO, 'Pre-trained model restored from %s' %
              ('-'.join([pretrained_model_ckpt_path, str(FLAGS.ckpt_step)])), Notify.ENDC)


def get_depth_and_prob_map(full_images, scaled_cams, depth_start, depth_interval):
    """ Computes depth and prob map. Inference mode depends on regularization choice and whether refinement is used """
    # depth map inference using 3DCNNs
    if FLAGS.regularization == '3DCNNs':
        depth_map, prob_map = inference_mem(
            full_images, scaled_cams, FLAGS.max_d, depth_start, depth_interval, FLAGS.network_mode, inverse_depth=FLAGS.inverse_depth)

        if FLAGS.refinement:
            ref_image = tf.squeeze(
                tf.slice(full_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
            depth_map, residual_depth_map = depth_refine(
                depth_map, ref_image, prob_map, FLAGS.max_d, depth_start, depth_interval, FLAGS.network_mode, FLAGS.refinement_network,
                True, upsample_depth=FLAGS.upsample_before_refinement, refine_with_confidence=FLAGS.refine_with_confidence)
            return depth_map, prob_map, residual_depth_map
    # depth map inference using GRU
    elif FLAGS.regularization == 'GRU':
        depth_map, prob_map = inference_winner_take_all(full_images, scaled_cams,
                                                        depth_num, depth_start, depth_end, network_mode=FLAGS.network_mode, reg_type='GRU', inverse_depth=FLAGS.inverse_depth)
    else:
        raise NotImplementedError
    return depth_map, prob_map, tf.no_op()


def write_output(output_dir, out_depth_map, out_prob_map, out_images, out_cams, out_full_cams, out_full_images, out_index, out_residual_depth_map=None):
    """ Writes the output from the network to disk """
    upsample = True if FLAGS.refinement == True and FLAGS.upsample_before_refinement == True else False
    out_depth_map = np.squeeze(out_depth_map)

    # If we upsampled depth map to input size, then we need to write the full sized cams, probs and ref image
    if upsample:
        out_ref_image = np.squeeze(out_full_images)
        out_ref_cam = np.squeeze(out_full_cams)
        out_prob_map = np.squeeze(out_prob_map)
        out_prob_map = scale_image(
            out_prob_map, 1.0 / FLAGS.sample_scale, 'nearest')
    else:
        out_ref_image = np.squeeze(out_images)
        out_ref_cam = np.squeeze(out_cams)
        out_prob_map = np.squeeze(out_prob_map)
    out_ref_image = np.squeeze(out_ref_image[0, :, :, :])
    out_ref_cam = np.squeeze(out_ref_cam[0, :, :, :])
    out_index = np.squeeze(out_index)

    # paths
    depth_map_path = os.path.join(
        output_dir, '{}_init.pfm'.format(out_index))
    prob_map_path = os.path.join(
        output_dir, '{}_prob.pfm'.format(out_index))
    out_ref_image_path = os.path.join(
        output_dir, '{}.jpg'.format(out_index))
    out_ref_cam_path = os.path.join(
        output_dir, '{}.txt'.format(out_index))
    # png outputs
    prob_png = os.path.join(
        output_dir, '{}_prob.png'.format(out_index))
    depth_png = os.path.join(
        output_dir, '{}_depth.png'.format(out_index))

    # save output
    write_pfm(depth_map_path, out_depth_map)
    write_pfm(prob_map_path, out_prob_map)

    # for png outputs
    write_depth_map(depth_png, out_depth_map,
                    visualization=FLAGS.visualize)
    write_confidence_map(prob_png, out_prob_map)
    write_reference_image(out_ref_image, out_ref_image_path)

    write_cam(out_ref_cam_path, out_ref_cam)
    if out_residual_depth_map is not None and FLAGS.visualize:
        residual_path = depth_png.replace('_depth', '_depth_residual')
        out_residual_depth_map = np.squeeze(out_residual_depth_map)
        write_residual_depth_map(out_residual_depth_map, residual_path)
        # Write out an unrefined version of depth map for comparison
        unrefined_depth_map = out_depth_map - out_residual_depth_map
        unrefined_path = depth_png.replace(
            '_depth', '_depth_unrefined_inverse')
        write_inverse_depth_map(unrefined_depth_map, unrefined_path)


def set_shapes(scaled_images, full_images, scaled_cams, full_cams):
    """ Reshapes tensors to prepare for input to network """
    scaled_images.set_shape(tf.TensorShape(
        [None, FLAGS.view_num, None, None, 3]))
    full_images.set_shape(tf.TensorShape(
        [None, FLAGS.view_num, None, None, 3]))
    scaled_cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
    full_cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
    depth_start = tf.reshape(
        tf.slice(scaled_cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    depth_interval = tf.reshape(
        tf.slice(scaled_cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    depth_num = tf.cast(
        tf.reshape(tf.slice(scaled_cams, [0, 0, 1, 3, 2], [1, 1, 1, 1, 1]), []), 'int32')
    depth_end = tf.reshape(
        tf.slice(scaled_cams, [0, 0, 1, 3, 3], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])

    return depth_start, depth_end, depth_interval, depth_num


def init_inference(input_dir, output_dir, width, height):
    """ Performs some basic initialization before the main inference method is run """
    if width and height:
        FLAGS.width, FLAGS.height = width, height
    logger.info('Computing depth maps with MVSNet. Using input width x height = {} x {}.'.format(
        FLAGS.width, FLAGS.height))
    if FLAGS.wandb:
        mu.initialize_wandb(FLAGS, project='mvsnet-inference')
    return setup_output_dir(input_dir, output_dir)

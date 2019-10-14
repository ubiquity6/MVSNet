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
import tensorflow as tf
from mvsnet.loss import *
from mvsnet.preprocess import *
from mvsnet.model import inference_mem, depth_refine, inference_winner_take_all
from mvsnet.mvs_data_generation.cluster_generator import ClusterGenerator
import mvsnet.utils as mu
from mvsnet.mvs_data_generation.utils import scale_image
logger = mu.setup_logger('predictlib')
sys.path.append("../")
tf.app.flags.DEFINE_bool('wandb', False,
                         """Whether or not to log inference results to wandb""")
tf.app.flags.DEFINE_string('run_name', None,
                           """A name to use for wandb logging""")

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

    mvs_iterator = mvs_set.make_initializable_iterator()
    # if tensorflow eager execution is enabled, use the tfe.iterator below instead. Useful for debugging.
    #mvs_iterator = tfe.Iterator(mvs_set)#mvs_set.make_initializable_iterator()
    return mvs_iterator, sample_size


def setup_output_dir(input_dir, output_dir):
    "Creates output dir for saving mvsnet output"
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'depths_mvsnet')
    mu.mkdir_p(output_dir)
    logger.info('Running inference on {}'.format(output_dir))
    return output_dir


def load_model(sess):
    """Load trained model for inference """
    if FLAGS.model_dir is not None:
        ckpt_path = mu.ckpt_path(FLAGS.model_dir, FLAGS.regularization, FLAGS.network_mode)
        restorer = tf.train.Saver(tf.global_variables())
        model_path = mu.model_path(ckpt_path, FLAGS.ckpt_step)
        restorer.restore(sess, model_path)
        logger.info('Pre-trained model restored from {}'.format(model_path))


def get_depth_and_prob_map(full_images, scaled_cams, depth_start, depth_interval):
    """ Computes depth and prob map. Inference mode depends on regularization choice and whether refinement is used """
    # depth map inference using 3DCNN
    if FLAGS.regularization == '3DCNN':
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





def write_output_slice(output_dir, out_depth_map, out_prob_map, out_images, out_cams, out_full_cams, out_full_images, out_index, out_residual_depth_map=None):
    """ Writes the output from the network to disk """
    upsample = True if FLAGS.refinement == True and FLAGS.upsample_before_refinement == True else False
    out_depth_map = np.squeeze(out_depth_map)
    mu.mkdir_p(output_dir)

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


def write_output(output_dir, out_depth_map_batch, out_prob_map_batch, out_images_batch, out_cams_batch, out_full_cams_batch, out_full_images_batch, out_index_batch, out_residual_depth_map_batch=None):
    start = time.time()
    for i in range(FLAGS.batch_size):
        dmap = out_depth_map_batch[i]
        pmap = out_prob_map_batch[i]
        images = out_images_batch[i]
        cams = out_cams_batch[i]
        full_cams = out_full_cams_batch[i]
        full_images = out_full_images_batch[i]
        index = out_index_batch[i]
        if out_residual_depth_map_batch is not None:
            residual = out_residual_depth_map_batch[i]
        else:
            residual = None
        write_output_slice(output_dir, dmap, pmap, images, cams, full_cams, full_images, index, residual)
    logger.info('Time to write prediction results: {:.3f} s'.format(time.time() - start))
        



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


def init_inference(input_dir, **kwargs):
    """ Performs some basic initialization before the main inference method is run """
    # The app flags can be set by passed in kwargs, but flag must already exist, otherwise this will raise an error
    for key, value in kwargs.items():
        setattr(FLAGS, key, value)

    logger.info('Computing depth maps with MVSNet. Using input width x height = {} x {}.'.format(
        FLAGS.width, FLAGS.height))
    if FLAGS.run_name is None:
        FLAGS.run_name = 'model={}_ckpt={}'.format(
            FLAGS.model_dir, FLAGS.ckpt_step)
    if FLAGS.wandb:
        mu.initialize_wandb(FLAGS, project='mvsnet-inference')

    log_flags()
    return setup_output_dir(input_dir, FLAGS.output_dir)

def log_flags():
    """ Logs all flags and their values to the console """
    logger.info('*** Logging all FLAGS ***')
    for flag in FLAGS:
        logger.info('{} = {}'.format(flag, getattr(FLAGS,flag)))


def get_header():
    header = 'model_dir, ckpt_step, loss, less_one, less_three, debug \n'
    return header


def header_exists(path):
    try:
        with open(path, 'r') as f:
            fl = f.readlines()
            if len(fl) > 0:
                if fl[0] == get_header():
                    return True
        return False
    except Exception as e:
        # If the above line fails its likely because file does not exist
        # in which case we will need to create the file and write the header
        return False


def ensure_header_exists(path):
    """ Ensures that the file at path has the correct header """
    if header_exists(path):
        return True
    else:
        # if header doesn't exist, we write a new file with the header
        with open(path, 'a+') as f:
            f.write(get_header())
        return True


def write_results(path, loss, less_one, less_three, debug):
    """ Writes test results to a file. If the file doesn't exist it is created """
    try:
        ensure_header_exists(path)
        with open(path, 'a+') as f:
            new_line = '{}, {}, {}, {}, {}, {} \n'.format(
                FLAGS.model_dir, FLAGS.ckpt_step, loss, less_one, less_three, debug)
            f.write(new_line)
    except Exception as e:
        logger.error('Failed to write results with exception {}'.format(e))
        pass  # While it is too bad if results fail to write, we don't want to stop the process over it

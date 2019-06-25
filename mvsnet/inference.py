#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Test script.
"""
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
from mvsnet.model import inference_mem, depth_refine, inference_winner_take_all
from mvsnet.preprocess import *
from mvsnet.cnn_wrapper.common import Notify
from mvsnet.mvs_data_generation.cluster_generator import ClusterGenerator
import mvsnet.utils as mu

logger = mu.setup_logger('mvsnet-inference')
sys.path.append("../")

# dataset parameters
tf.app.flags.DEFINE_string('input_dir', None,
                           """Path to data to run inference on""")
tf.app.flags.DEFINE_string('output_dir', None,
                           """Path to data to dir to output results""")
tf.app.flags.DEFINE_string('model_dir',
                           'gs://mvs-training-mlengine/mvsnet_refine_geom_viewnum7_conf_refine_2/models/',
                           """Path to restore the model.""")
tf.app.flags.DEFINE_integer('ckpt_step', 110000,
                            """ckpt  step.""")
# input parameters
tf.app.flags.DEFINE_integer('view_num', 6,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 192,
                            """Maximum depth step when testing.""")
tf.app.flags.DEFINE_integer('width', 512,
                            """Maximum image width when testing.""")
tf.app.flags.DEFINE_integer('height', 384,
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
tf.app.flags.DEFINE_boolean('refinement', True,
                            """Whether to apply depth map refinement for MVSNet""")
tf.app.flags.DEFINE_bool('inverse_depth', True,
                         """Whether to apply inverse depth for R-MVSNet""")
tf.app.flags.DEFINE_string('network_mode', 'normal',
                           """One of 'normal', 'lite' or 'ultralite'. If 'lite' or 'ultralite' then networks have fewer params""")
tf.app.flags.DEFINE_string('refinement_network', 'original',
                           """Specifies network to use for refinement. One of 'original' or 'unet'. 
                            If 'original' then the original mvsnet refinement network is used, otherwise a unet style architecture is used.""")
tf.app.flags.DEFINE_boolean('upsample_before_refinement', True,
                            """Whether to upsample depth map to input resolution before the refinement network.""")
tf.app.flags.DEFINE_boolean('refine_with_confidence', True,
                            """Whether or not to concatenate the confidence map as an input channel to refinement network""")
FLAGS = tf.app.flags.FLAGS


def setup_data_iterator(input_dir):
    # testing set
    data_gen = ClusterGenerator(input_dir, FLAGS.view_num, FLAGS.width, FLAGS.height,
                                FLAGS.max_d, FLAGS.interval_scale, FLAGS.base_image_size, mode='test')
    mvs_generator = iter(data_gen)
    sample_size = len(data_gen.train_clusters)

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
    # create output folder
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'depths_mvsnet')
    mu.mkdir_p(output_dir)
    return output_dir


def get_depth_end(scaled_cams, depth_start, depth_num, depth_interval):
    # deal with inverse depth
    if FLAGS.regularization == '3DCNNs' and FLAGS.inverse_depth:
        depth_end = tf.reshape(
            tf.slice(scaled_cams, [0, 0, 1, 3, 3], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    else:
        depth_end = depth_start + \
            (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    return depth_end


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


def get_depth_and_prob_map(centered_images, scaled_cams, depth_start, depth_interval):
    """ Computes depth and prob map. Inference mode depends on regularization choice and whether refinement is used """
    # depth map inference using 3DCNNs
    if FLAGS.regularization == '3DCNNs':
        init_depth_map, prob_map = inference_mem(
            centered_images, scaled_cams, FLAGS.max_d, depth_start, depth_interval, FLAGS.network_mode)

        if FLAGS.refinement:
            ref_image = tf.squeeze(
                tf.slice(centered_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
            init_depth_map = depth_refine(
                init_depth_map, ref_image, prob_map, FLAGS.max_d, depth_start, depth_interval, FLAGS.network_mode, FLAGS.refinement_network,
                True, upsample_depth=FLAGS.upsample_before_refinement, refine_with_confidence=FLAGS.refine_with_confidence)
    # depth map inference using GRU
    elif FLAGS.regularization == 'GRU':
        init_depth_map, prob_map = inference_winner_take_all(centered_images, scaled_cams,
                                                             depth_num, depth_start, depth_end, network_mode=FLAGS.network_mode, reg_type='GRU', inverse_depth=FLAGS.inverse_depth, training=False)
    else:
        raise NotImplementedError
    return init_depth_map, prob_map


def get_sample


def compute_depth_maps(input_dir, output_dir=None, width=None, height=None):
    """ Performs inference using trained MVSNet model on data located in input_dir """
    if width and height:
        FLAGS.width, FLAGS.height = width, height
    logger.info('Computing depth maps with MVSNet. Using input width x height = {} x {}.\n Flags: {}'.format(
        FLAGS.width, FLAGS.height, FLAGS))

    output_dir = setup_output_dir(input_dir, output_dir)
    mvs_iterator, sample_size = setup_data_iterator(input_dir)

    scaled_images, centered_images, scaled_cams, full_cams, image_index = mvs_iterator.get_next()

    # set shapes
    scaled_images.set_shape(tf.TensorShape(
        [None, FLAGS.view_num, None, None, 3]))
    centered_images.set_shape(tf.TensorShape(
        [None, FLAGS.view_num, None, None, 3]))
    scaled_cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
    full_cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
    depth_start = tf.reshape(
        tf.slice(scaled_cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    depth_interval = tf.reshape(
        tf.slice(scaled_cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    depth_num = tf.cast(
        tf.reshape(tf.slice(scaled_cams, [0, 0, 1, 3, 2], [1, 1, 1, 1, 1]), []), 'int32')

    depth_end = get_depth_end(scaled_cams, depth_start,
                              depth_num, depth_interval)

    init_depth_map, prob_map = get_depth_and_prob_map(
        centered_images, scaled_cams, depth_start, depth_interval)

    # init option
    var_init_op = tf.local_variables_initializer()
    init_op, config = mu.init_session()

    with tf.Session(config=config) as sess:
        # initialization
        sess.run(var_init_op)
        sess.run(init_op)

        load_model(sess)
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
            write_pfm(init_depth_map_path, out_init_depth_image)
            write_pfm(prob_map_path, out_prob_map)

            # for png outputs
            write_depth_map(depth_png, out_init_depth_image,
                            visualization=True)
            write_confidence_map(prob_png, out_prob_map)

            out_ref_image = cv2.cvtColor(out_ref_image, cv2.COLOR_RGB2BGR)
            image_file = file_io.FileIO(out_ref_image_path, mode='w')
            scipy.misc.imsave(image_file, out_ref_image)
            write_cam(out_ref_cam_path, out_ref_cam)


def main(_):  # pylint: disable=unused-argument
    """
    Program entrance for running inference with MVSNet
    Acceptable input for the input_dir are (1) a single test folder, or (2) a folder containing multiple
    test folders. We check to see which one it is
    """
    if os.path.isfile(os.path.join(FLAGS.input_dir, 'covisibility.json')):
        compute_depth_maps(FLAGS.input_dir)
    else:
        folders = os.listdir(FLAGS.input_dir)
        for f in folders:
            compute_depth_maps(os.path.join(FLAGS.input_dir, f))


if __name__ == '__main__':
    tf.app.run()

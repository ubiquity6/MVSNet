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
import wandb
import tensorflow as tf
from mvsnet.loss import *
from mvsnet.model import inference_mem, depth_refine, inference_winner_take_all
from mvsnet.preprocess import *
from mvsnet.cnn_wrapper.common import Notify
from mvsnet.mvs_data_generation.cluster_generator import ClusterGenerator
import mvsnet.utils as mu
from mvsnet.mvs_data_generation.utils import scale_image

logger = mu.setup_logger('mvsnet-inference')
sys.path.append("../")

# dataset parameters
tf.app.flags.DEFINE_string('input_dir', None,
                           """Path to data to run inference on""")
tf.app.flags.DEFINE_string('output_dir', None,
                           """Path to data to dir to output results""")
tf.app.flags.DEFINE_string('model_dir',
                           None,
                           """Path to restore the model.""")
tf.app.flags.DEFINE_integer('ckpt_step', None,
                            """ckpt  step.""")
tf.app.flags.DEFINE_string('run_name', None,
                           """A name to use for wandb logging""")
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
tf.app.flags.DEFINE_boolean('refinement', None,
                            """Whether to apply depth map refinement for MVSNet""")
tf.app.flags.DEFINE_bool('inverse_depth', False,
                         """Whether to apply inverse depth for R-MVSNet""")
tf.app.flags.DEFINE_string('network_mode', 'normal',
                           """One of 'normal', 'lite' or 'ultralite'. If 'lite' or 'ultralite' then networks have fewer params""")
tf.app.flags.DEFINE_string('refinement_network', None,
                           """Specifies network to use for refinement. One of 'original' or 'unet'.
                            If 'original' then the original mvsnet refinement network is used, otherwise a unet style architecture is used.""")
tf.app.flags.DEFINE_boolean('upsample_before_refinement', True,
                            """Whether to upsample depth map to input resolution before the refinement network.""")
tf.app.flags.DEFINE_boolean('refine_with_confidence', True,
                            """Whether or not to concatenate the confidence map as an input channel to refinement network""")

# Parameters for writing and benchmarking output
tf.app.flags.DEFINE_bool('visualize', True,
                         """If visualize is true, the inference script will write some auxiliary files for visualization and debugging purposes.
                         This is useful when developing and debugging, but should probably be turned off in production""")
tf.app.flags.DEFINE_bool('wandb', False,
                         """Whether or not to log inference results to wandb""")
tf.app.flags.DEFINE_bool('benchmark', True,
                         """If benchmark is True, the network results will be benchmarked against GT.
                         This should only be used if the input_dir contains GT depth maps""")
tf.app.flags.DEFINE_bool('write_output', False,
                         """When benchmarking you can set this to False if you don't need the output""")
tf.app.flags.DEFINE_bool('reuse_vars', False,
                         """A global flag representing whether variables should be reused. This should be
                          set to False by default and is switched on or off by individual methods""")
tf.app.flags.DEFINE_integer('max_clusters_per_session', 4,
                            """The maximum number of clusters to benchmark per session. If not benchmarking this should probably be set to None""")
FLAGS = tf.app.flags.FLAGS


def setup_data_iterator(input_dir):
    "Configures the data generator that is used to feed batches of data for inference"
    mode = 'benchmark' if FLAGS.benchmark else 'test'
    print(mode)
    data_gen = ClusterGenerator(input_dir, FLAGS.view_num, FLAGS.width, FLAGS.height, FLAGS.max_d, FLAGS.interval_scale, \
                FLAGS.base_image_size, mode = mode, val_split = 0.0,  benchmark = FLAGS.benchmark, output_scale = FLAGS.sample_scale, max_clusters_per_session = FLAGS.max_clusters_per_session)
    mvs_generator=iter(data_gen)
    sample_size=len(data_gen.train_clusters)

    if FLAGS.benchmark:
        generator_data_type=(tf.float32, tf.float32,
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


def compute_depth_maps(input_dir, output_dir=None, width=None, height=None):
    """ Performs inference using trained MVSNet model on data located in input_dir and writes data to disk"""
    FLAGS.benchmark = False
    output_dir = init_inference(input_dir, output_dir, width, height)
    mvs_iterator, sample_size = setup_data_iterator(input_dir)
    scaled_images, full_images, scaled_cams, full_cams, image_index = mvs_iterator.get_next()

    depth_start, depth_end, depth_interval, depth_num = set_shapes(
        scaled_images, full_images, scaled_cams, full_cams)

    depth_map, prob_map, residual_depth_map = get_depth_and_prob_map(
        full_images, scaled_cams, depth_start, depth_interval)

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
            out_residual_depth_map = None
            fetches = [depth_map, prob_map, scaled_images,
                       scaled_cams, full_cams, full_images, image_index, residual_depth_map]
            try:
                out_depth_map, out_prob_map, out_images, out_cams, out_full_cams, out_full_images, out_index, out_residual_depth_map = sess.run(
                    fetches)
            except tf.errors.OutOfRangeError:
                logger.info("all dense finished")  # ==> "End of dataset"
                break
            print(Notify.INFO, 'depth inference %d/%d finished. Image index %d. (%.3f sec/step)' % (step, sample_size, out_index, time.time() - start_time),
                  Notify.ENDC)
            write_output(output_dir, out_depth_map, out_prob_map, out_images,
                         out_cams, out_full_cams, out_full_images, out_index, out_residual_depth_map)


def benchmark_depth_maps(input_dir, losses, less_ones, less_threes, output_dir=None, width=None, height=None):
    """ Performs inference using trained MVSNet model on data located in input_dir. This method is similar to compute_depth_maps, however it benchmarks the resulting 
    data against GT depths. This is useful for benchmarking models and should only be run on input data with GT depth maps  """
    FLAGS.benchmark = True
    output_dir = init_inference(input_dir, output_dir, width, height)
    mvs_iterator, sample_size = setup_data_iterator(input_dir)
    scaled_images, full_images, scaled_cams, full_cams, full_depth, image_index, session_dir = mvs_iterator.get_next()

    depth_start, depth_end, depth_interval, depth_num = set_shapes(
        scaled_images, full_images, scaled_cams, full_cams)

    depth_map, prob_map, residual_depth_map = get_depth_and_prob_map(
        full_images, scaled_cams, depth_start, depth_interval)
    out_residual_depth_map = None
    full_depth_shape = tf.shape(full_depth)
    upsample_depth = False if FLAGS.refinement == True and FLAGS.upsample_before_refinement == True else True
    if upsample_depth:
        depth_map = tf.image.resize_bilinear(
            depth_map, [full_depth_shape[1], full_depth_shape[2]])
    loss, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
        depth_map, full_depth, depth_start, depth_end)

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
                if FLAGS.refinement:
                    out_depth_map, out_prob_map, out_images, out_cams, out_full_cams, out_full_images, out_index, out_session_dir, out_loss, out_less_one, out_less_three, out_residual_depth_map = sess.run(
                        [depth_map, prob_map, scaled_images, scaled_cams, full_cams, full_images, image_index, session_dir, loss, less_one_accuracy, less_three_accuracy, residual_depth_map])
                else:
                    out_depth_map, out_prob_map, out_images, out_cams, out_full_cams, out_full_images, out_index, out_loss, out_less_one, out_less_three = sess.run(
                        [depth_map, prob_map, scaled_images, scaled_cams, full_cams, full_images, image_index, session_dir, loss, less_one_accuracy, less_three_accuracy])
            except tf.errors.OutOfRangeError:
                print("all dense finished")  # ==> "End of dataset"
                break
            print(Notify.INFO, 'depth inference %d/%d finished. Image index %d. (%.3f sec/step)' % (step, sample_size, out_index, time.time() - start_time),
                  Notify.ENDC)
            logger.debug(
                'Performed inference for reference image {}'.format(out_index))
            logger.info('Image {} loss = {}'.format(
                out_index, out_loss))
            logger.info('Image {} less one = {}'.format(
                out_index, out_less_one))
            logger.info('Image {} less three = {}'.format(
                out_index, out_less_three))

            write_dir = os.path.join(str(out_session_dir[0]), 'depths_mvsnet') 
            mu.mkdir_p(write_dir)
            if FLAGS.write_output:
                write_output(write_dir, out_depth_map, out_prob_map, out_images,
                            out_cams, out_full_cams, out_full_images, out_index, out_residual_depth_map)
            losses.append(out_loss)
            less_ones.append(out_less_one)
            less_threes.append(out_less_three)
            if FLAGS.wandb:
                wandb.log(
                    {'loss': out_loss, 'less_three': out_less_three, 'less_one': out_less_one})


def main(_):  # pylint: disable=unused-argument
    """
    Program entrance for running inference with MVSNet
    Acceptable input for the input_dir are (1) a single test folder, or (2) a folder containing multiple
    test folders. We check to see which one it is
    """
    run_dir = os.path.isfile(os.path.join(
        FLAGS.input_dir, 'covisibility.json'))
    sub_dirs = [f for f in tf.gfile.ListDirectory(
                FLAGS.input_dir) if not f.startswith('.') if not f.endswith('.txt')]
    if FLAGS.benchmark:
        losses = []
        less_ones = []
        less_threes = []
        benchmark_depth_maps(FLAGS.input_dir, losses,
                                less_ones, less_threes)
        avg_loss = np.asarray(losses).mean()
        avg_less_one = np.asarray(less_ones).mean()
        avg_less_three = np.asarray(less_threes).mean()
        logger.info(' ** Average Loss = {}'.format(avg_loss))
        logger.info(
            ' ** Average Less one = {}'.format(avg_less_one))
        logger.info(
            ' ** Average Less three = {}'.format(avg_less_three))
        if FLAGS.wandb:
            wandb.log(
                {'avg_loss': avg_loss, 'avg_less_three': avg_less_three, 'avg_less_one': avg_less_one})
    else:
        if run_dir:
            compute_depth_maps(FLAGS.input_dir)
        else:
            for f in sub_dirs:
                data_dir = os.path.join(
                    FLAGS.input_dir, f)
                logger.info('Computing depth maps on dir {}'.format(data_dir))
                compute_depth_maps(data_dir)
                tf.app.flags.FLAGS.reuse_vars = True


if __name__ == '__main__':
    tf.app.run()

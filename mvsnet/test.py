#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Test script.
"""
from __future__ import print_function
import os
import time
import sys
import wandb
import tensorflow as tf
import numpy as np
from mvsnet.cnn_wrapper.common import Notify
from mvsnet.loss import mvsnet_regression_loss
from mvsnet.mvs_data_generation.utils import scale_and_reshape_depth
import mvsnet.utils as mu
import mvsnet.predictlib as pl

logger = mu.setup_logger('mvsnet-test')
sys.path.append("../")

# dataset parameters
tf.app.flags.DEFINE_string('input_dir', None,
                           """Path to data to run inference on""")
tf.app.flags.DEFINE_string('output_dir', 'depths_mvsnet',
                           """Name of directory to write output to. Base path is root of session directory""")
tf.app.flags.DEFINE_string('model_dir',
                           'gs://mvs-training-mlengine/trained-models/08-19-2019/',
                           """Path to restore the model.""")
tf.app.flags.DEFINE_integer('ckpt_step', 400000,
                            """ckpt  step.""")
# input parameters
tf.app.flags.DEFINE_integer('view_num', 4,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 192,
                            """Maximum depth step when testing.""")
tf.app.flags.DEFINE_integer('width', 640,
                            """Maximum image width when testing.""")
tf.app.flags.DEFINE_integer('height', 512,
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
tf.app.flags.DEFINE_string('regularization', '3DCNN',
                           """Regularization method, including '3DCNN' and 'GRU'""")
tf.app.flags.DEFINE_boolean('refinement', False,
                            """Whether to apply depth map refinement for MVSNet""")
tf.app.flags.DEFINE_bool('inverse_depth', False,
                         """Whether to apply inverse depth for R-MVSNet""")
tf.app.flags.DEFINE_string('network_mode', 'normal',
                           """One of 'normal', 'lite' or 'ultralite'. If 'lite' or 'ultralite' then networks have fewer params""")
tf.app.flags.DEFINE_string('refinement_network', 'unet',
                           """Specifies network to use for refinement. One of 'original' or 'unet'.
                            If 'original' then the original mvsnet refinement network is used, otherwise a unet style architecture is used.""")
tf.app.flags.DEFINE_boolean('upsample_before_refinement', True,
                            """Whether to upsample depth map to input resolution before the refinement network.""")
tf.app.flags.DEFINE_boolean('refine_with_confidence', True,
                            """Whether or not to concatenate the confidence map as an input channel to refinement network""")
tf.app.flags.DEFINE_bool('grad_loss', True,
                         """Whether or not to add a depth gradient term to the overall loss""")

# Parameters for writing and benchmarking output
tf.app.flags.DEFINE_bool('visualize', False,
                         """If visualize is true, the inference script will write some auxiliary files for visualization and debugging purposes.
                         This is useful when developing and debugging, but should probably be turned off in production""")
tf.app.flags.DEFINE_bool('benchmark', True,
                         """If benchmark is True, the network results will be benchmarked against GT.
                         This should only be used if the input_dir contains GT depth maps""")
tf.app.flags.DEFINE_bool('write_output', False,
                         """When benchmarking you can set this to False if you don't need the output""")
tf.app.flags.DEFINE_bool('reuse_vars', False,
                         """A global flag representing whether variables should be reused. This should be
                          set to False by default and is switched on or off by individual methods""")
tf.app.flags.DEFINE_integer('max_clusters_per_session', 100,
                            """The maximum number of clusters to benchmark per session. If not benchmarking this should probably be set to None""")
tf.app.flags.DEFINE_string('results_path', './results.csv',
                           """Path to csv file to write results to""")
FLAGS = tf.app.flags.FLAGS


def benchmark_depth_maps(input_dir, losses, less_ones, less_threes, out_debugs, **kwargs):
    """ Performs inference using trained MVSNet model on data located in input_dir. This method is similar to compute_depth_maps, however it benchmarks the resulting 
    data against GT depths. This is useful for benchmarking models and should only be run on input data with GT depth maps  """
    output_dir = pl.init_inference(input_dir, **kwargs)
    mvs_iterator, sample_size = pl.setup_data_iterator(input_dir)
    scaled_images, full_images, scaled_cams, full_cams, full_depth, image_index, session_dir = mvs_iterator.get_next()

    depth_start, depth_end, depth_interval, depth_num = pl.set_shapes(
        scaled_images, full_images, scaled_cams, full_cams)

    depth_map, prob_map, residual_depth_map = pl.get_depth_and_prob_map(
        full_images, scaled_cams, depth_start, depth_interval)
    out_residual_depth_map = None
    full_depth_shape = tf.shape(full_depth)
    upsample_depth = True if FLAGS.refinement == True and FLAGS.upsample_before_refinement == True else False
    if upsample_depth:
        depth_map = tf.image.resize_bilinear(
            depth_map, [full_depth_shape[1], full_depth_shape[2]])
    loss, less_one_accuracy, less_three_accuracy, debug = mvsnet_regression_loss(
        depth_map, full_depth, depth_start, depth_end, grad_loss=FLAGS.grad_loss)

    # init option
    var_init_op = tf.local_variables_initializer()
    init_op, config = mu.init_session()
    out_residual_depth_map = None

    with tf.Session(config=config) as sess:
        # initialization
        sess.run(var_init_op)
        sess.run(init_op)
        pl.load_model(sess)
        sess.run(mvs_iterator.initializer)
        for step in range(sample_size):
            start_time = time.time()
            try:
                if FLAGS.refinement:
                    out_depth_map, out_prob_map, out_images, out_cams, out_full_cams, out_full_images, out_index, out_session_dir, out_loss, out_less_one, out_less_three, out_residual_depth_map, out_debug = sess.run(
                        [depth_map, prob_map, scaled_images, scaled_cams, full_cams, full_images, image_index, session_dir, loss, less_one_accuracy, less_three_accuracy, residual_depth_map, debug])
                else:
                    out_depth_map, out_prob_map, out_images, out_cams, out_full_cams, out_full_images, out_index, out_session_dir, out_loss, out_less_one, out_less_three, out_debug = sess.run(
                        [depth_map, prob_map, scaled_images, scaled_cams, full_cams, full_images, image_index, session_dir, loss, less_one_accuracy, less_three_accuracy, debug])
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
            logger.info('Image {} debug = {}'.format(
                out_index, out_debug))

            write_dir = os.path.join(str(out_session_dir[0]), FLAGS.output_dir)
            mu.mkdir_p(write_dir)
            if FLAGS.write_output:
                pl.write_output(write_dir, out_depth_map, out_prob_map, out_images,
                                out_cams, out_full_cams, out_full_images, out_index, out_residual_depth_map)
            losses.append(out_loss)
            less_ones.append(out_less_one)
            less_threes.append(out_less_three)
            out_debugs.append(out_debug)
            if FLAGS.wandb:
                wandb.log(
                    {'loss': out_loss, 'less_three': out_less_three, 'less_one': out_less_one, 'debug': out_debug})


def main(_):  # pylint: disable=unused-argument
    """
    Program entrance for running inference with MVSNet
    Acceptable input for the input_dir are (1) a single test folder, or (2) a folder containing multiple
    test folders. We check to see which one it is
    """
    losses = []
    less_ones = []
    less_threes = []
    out_debugs = []
    benchmark_depth_maps(FLAGS.input_dir, losses,
                         less_ones, less_threes, out_debugs)
    avg_loss = np.asarray(losses).mean()
    avg_less_one = np.asarray(less_ones).mean()
    avg_less_three = np.asarray(less_threes).mean()
    avg_debug = np.asarray(out_debugs).mean()
    logger.info(' ** Average Loss = {}'.format(avg_loss))
    logger.info(
        ' ** Average Less one = {}'.format(avg_less_one))
    logger.info(
        ' ** Average Less three = {}'.format(avg_less_three))
    logger.info(
        ' ** Average debug = {}'.format(avg_debug))
    pl.write_results(FLAGS.results_path, avg_loss,
                     avg_less_one, avg_less_three, avg_debug)

    if FLAGS.wandb:
        wandb.log(
            {'avg_loss': avg_loss, 'avg_less_three': avg_less_three, 'avg_less_one': avg_less_one})


if __name__ == '__main__':
    tf.app.run()

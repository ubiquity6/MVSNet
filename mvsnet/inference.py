#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Test script.
"""
from __future__ import print_function
import os
import time
import sys
import tensorflow as tf
from mvsnet.cnn_wrapper.common import Notify
import mvsnet.utils as mu
import mvsnet.predictlib as pl

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
# input parameters
tf.app.flags.DEFINE_integer('view_num', 10,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 256,
                            """Maximum depth step when testing.""")
tf.app.flags.DEFINE_integer('width', 1024,
                            """Maximum image width when testing.""")
tf.app.flags.DEFINE_integer('height', 768,
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
tf.app.flags.DEFINE_bool('inverse_depth', False,
                         """Whether to apply inverse depth for R-MVSNet""")
tf.app.flags.DEFINE_string('network_mode', 'normal',
                           """One of 'normal', 'lite' or 'ultralite'. If 'lite' or 'ultralite' then networks have fewer params""")
tf.app.flags.DEFINE_string('refinement_network', 'unet',
                           """Specifies network to use for refinement. One of 'original' or 'unet'.
                            If 'original' then the original mvsnet refinement network is used, otherwise a unet style architecture is used.""")
tf.app.flags.DEFINE_boolean('upsample_before_refinement', False,
                            """Whether to upsample depth map to input resolution before the refinement network.""")
tf.app.flags.DEFINE_boolean('refine_with_confidence', True,
                            """Whether or not to concatenate the confidence map as an input channel to refinement network""")

# Parameters for writing and benchmarking output
tf.app.flags.DEFINE_bool('visualize', False,
                         """If visualize is true, the inference script will write some auxiliary files for visualization and debugging purposes.
                         This is useful when developing and debugging, but should probably be turned off in production""")
tf.app.flags.DEFINE_bool('benchmark', False,
                         """If benchmark is True, the network results will be benchmarked against GT.
                         This should only be used if the input_dir contains GT depth maps""")
tf.app.flags.DEFINE_bool('write_output', True,
                         """When benchmarking you can set this to False if you don't need the output""")
tf.app.flags.DEFINE_bool('reuse_vars', False,
                         """A global flag representing whether variables should be reused. This should be
                          set to False by default and is switched on or off by individual methods""")
tf.app.flags.DEFINE_integer('max_clusters_per_session', None,
                            """The maximum number of clusters to benchmark per session. If not benchmarking this should probably be set to None""")

FLAGS = tf.app.flags.FLAGS


def compute_depth_maps(input_dir, output_dir=None, width=None, height=None):
    """ Performs inference using trained MVSNet model on data located in input_dir and writes data to disk"""
    output_dir = pl.init_inference(input_dir, output_dir, width, height)
    mvs_iterator, sample_size = pl.setup_data_iterator(input_dir)
    scaled_images, full_images, scaled_cams, full_cams, image_index = mvs_iterator.get_next()

    depth_start, depth_end, depth_interval, depth_num = pl.set_shapes(
        scaled_images, full_images, scaled_cams, full_cams)

    depth_map, prob_map, residual_depth_map = pl.get_depth_and_prob_map(
        full_images, scaled_cams, depth_start, depth_interval)

    # init option
    var_init_op = tf.local_variables_initializer()
    init_op, config = mu.init_session()
    with tf.Session(config=config) as sess:
        # initialization
        sess.run(var_init_op)
        sess.run(init_op)
        pl.load_model(sess)
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
            pl.write_output(output_dir, out_depth_map, out_prob_map, out_images,
                            out_cams, out_full_cams, out_full_images, out_index, out_residual_depth_map)


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
    if run_dir:
        compute_depth_maps(FLAGS.input_dir)
    else:
        for f in sub_dirs:
            data_dir = os.path.join(
                FLAGS.input_dir, f)
            logger.info('Computing depth maps on dir {}'.format(data_dir))
            compute_depth_maps(data_dir)
            # By setting reuse_vars = True this ensures that the second time compute_depth_maps
            # is run that the computational graph is not re-initialized
            tf.app.flags.FLAGS.reuse_vars = True


if __name__ == '__main__':
    tf.app.run()

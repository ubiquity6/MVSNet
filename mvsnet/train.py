#!/usr/bin/env python
from __future__ import print_function

from mvsnet.homography_warping import get_homographies, homography_warping
from mvsnet.model import *
from mvsnet.preprocess import *
from mvsnet.loss import *
from mvsnet.cnn_wrapper.common import Notify
from mvsnet.mvs_data_generation.cluster_generator import ClusterGenerator
from mvsnet.mvs_data_generation import utils as ut
import mvsnet.utils as mu
"""
Copyright 2019, Yao Yao, HKUST.
Training script.
"""

import os
import time
import sys
import math
import argparse
from random import randint
import cv2
import numpy as np
import wandb
import tensorflow as tf
import subprocess
from tensorflow.python.lib.io import file_io


logger = mu.setup_logger('mvsnet-train')

# params for datasets
tf.app.flags.DEFINE_string('train_data_root', None,
                           """Path to dtu dataset.""")
tf.app.flags.DEFINE_string('logs_dir', None,
                           """Path to store the log.""")
tf.app.flags.DEFINE_string('model_dir', None,
                           """Path to save the model.""")
tf.app.flags.DEFINE_string('model_load_dir', None,
                           """Path to load the saved model. If not specified, model will be loaded from model_dir""")
tf.app.flags.DEFINE_string('job-dir', None,
                           """Path to save job artifacts""")
tf.app.flags.DEFINE_boolean('train_dtu', True,
                            """Whether to train.""")
tf.app.flags.DEFINE_boolean('use_pretrain', True,
                            """Whether to train.""")
tf.app.flags.DEFINE_integer('ckpt_step', None,
                            """ckpt step.""")
tf.app.flags.DEFINE_string('run_name', None,
                           """A name to use for wandb logging""")

# input parameters
tf.app.flags.DEFINE_integer('view_num', 3,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 128,
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('width', 512,
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('height', 384,
                            """Maximum image height when training.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25,
                          """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_float('interval_scale', 1.0,
                          """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_float('base_image_size', 8,
                          """Base image size""")
tf.app.flags.DEFINE_bool('inverse_depth', False,
                         """Whether to apply inverse depth for R-MVSNet""")
# network architectures
tf.app.flags.DEFINE_string('regularization', '3DCNNs',
                           """Regularization method.""")
tf.app.flags.DEFINE_string('optimizer', 'rmsprop',
                           """Optimizer to use. One of 'momentum', 'rmsprop' or 'adam' """)
tf.app.flags.DEFINE_boolean('refinement', True,
                            """Whether to apply depth map refinement for 3DCNNs""")
tf.app.flags.DEFINE_string('refinement_train_mode', 'refine_only',
                            """One of 'all', 'refine_only' or 'main_only'. If 'main_only' then only the main network is trained,
                            if 'refine_only', only the refinement network is trained, and if 'all' then the whole network is trained.
                            Note this is only applicable if training with refinement=True and 3DCNN regularization """)
tf.app.flags.DEFINE_string('network_mode', 'normal',
                            """One of 'normal', 'lite' or 'ultralite'. If 'lite' or 'ultralite' then networks have fewer params""")
tf.app.flags.DEFINE_string('refinement_network', 'unet',
                            """Specifies network to use for refinement. One of 'original' or 'unet'. 
                            If 'original' then the original mvsnet refinement network is used, otherwise a unet style architecture is used.""")
tf.app.flags.DEFINE_boolean('upsample_before_refinement', True,
                            """Whether to upsample depth map to input resolution before the refinement network""")
tf.app.flags.DEFINE_boolean('refine_with_confidence', True,
                            """Whether or not to concatenate the confidence map as an input channel to refinement network""")
tf.app.flags.DEFINE_boolean('refine_with_stereo', False,
                            """Whether or not to inject a stereo partner into refinement network""")
# training parameters
tf.app.flags.DEFINE_integer('num_gpus', None,
                            """Number of GPUs.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Training batch size.""")
tf.app.flags.DEFINE_integer('epoch', None,
                            """Training epoch number.""")
tf.app.flags.DEFINE_float('base_lr', 0.001,
                          """Base learning rate.""")
tf.app.flags.DEFINE_integer('display', 1,
                            """Interval of loginfo display.""")
tf.app.flags.DEFINE_integer('stepvalue', 75000,
                            """Step interval to decay learning rate.""")
tf.app.flags.DEFINE_integer('snapshot', 5000,
                            """Step interval to save the model.""")
tf.app.flags.DEFINE_float('gamma', 0.5,
                          """Learning rate decay rate.""")
tf.app.flags.DEFINE_float('val_batch_size', 50,
                          """Number of images to run validation on when validation.""")
tf.app.flags.DEFINE_float('train_steps_per_val', 500,
                          """Number of samples to train on before running a round of validation.""")
tf.app.flags.DEFINE_float('dataset_fraction', 1.0,
                          """Fraction of dataset to use for training. Float between 0 and 1. NOTE: For training a production model
                           you should use 1, but for experiments it may be useful to use a fraction less than 1.""")
tf.app.flags.DEFINE_float('decay_per_10_epoch', 0.01,
                          """ The fraction by which learning rate should decay every 10 epochs""")
tf.app.flags.DEFINE_bool('wandb', True,
                         """Whether or not to log results to wandb""")
tf.app.flags.DEFINE_bool('reuse_vars', False,
                         """A global flag representing whether variables should be reused. This should be 
                          set to False by default and is switched on or off by individual methods""")

FLAGS = tf.app.flags.FLAGS

def load_model(sess):
    """ Loads pretrained model if supplied  """
    total_step = 0
    if FLAGS.ckpt_step:
        if FLAGS.model_load_dir:
            pretrained_model_path = os.path.join(
                FLAGS.model_load_dir, FLAGS.regularization, 'model.ckpt')
        else:
            pretrained_model_path = os.path.join(
                FLAGS.model_dir, FLAGS.regularization, 'model.ckpt')
        restorer = tf.train.Saver(tf.global_variables())
        restorer.restore(
            sess, '-'.join([pretrained_model_path, str(FLAGS.ckpt_step)]))
        print(Notify.INFO, 'Pre-trained model restored from %s' %
                ('-'.join([pretrained_model_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
        total_step = FLAGS.ckpt_step
    return total_step

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def generator(n, mode):
    """ Returns a data generator object.
    Args:
        mode: One of 'training' or 'validation'
        """
    flip_cams = False
    if FLAGS.regularization == 'GRU':
        flip_cams = True
    gen = ClusterGenerator(FLAGS.train_data_root, FLAGS.view_num, FLAGS.width, FLAGS.height,
                                FLAGS.max_d, FLAGS.interval_scale, FLAGS.base_image_size, mode=mode, flip_cams=flip_cams, sessions_frac = FLAGS.dataset_fraction)
    logger.info('Initializing generator with mode {}'.format(mode))
    if mode == 'training':
        global training_sample_size
        training_sample_size = len(gen.train_clusters)
        if FLAGS.regularization == 'GRU':
            training_sample_size = training_sample_size * 2
        
    return iter(gen)


def training_dataset(n):
    """ Returns a dataset over the Training data, initialized from a data generator
    """
    generator_data_type = (tf.float32, tf.float32, tf.float32, tf.float32)
    training_set = tf.data.Dataset.from_generator(
        lambda: generator(n, mode='training'), generator_data_type)
    training_set = training_set.batch(FLAGS.batch_size)
    training_set = training_set.prefetch(buffer_size=1)
    return training_set


def validation_dataset(n):
    """ Returns a dataset over the Validation data, initialized from a data generator
    """
    generator_data_type = (tf.float32, tf.float32, tf.float32, tf.float32)
    validation_set = tf.data.Dataset.from_generator(
        lambda: generator(n, mode='validation'), generator_data_type)
    validation_set = validation_set.batch(FLAGS.batch_size)
    validation_set = validation_set.prefetch(buffer_size=1)
    return validation_set

def parallel_iterator(mode, num_generators = FLAGS.num_gpus):
    """ The parallel iterator returns a datagenerator that is parallelized by
    interleaving multiple data generators. This uses the Tensorflow parallel_interleave
    feature: https: // www.tensorflow.org/api_docs/python/tf/data/experimental/parallel_interleave
    Args:
        mode: Whether this is for 'training' or 'testing'
        num_generators: How many parallel generators to use(equivalent to num parallel threads)
    Returns:
        Tensorflow Dataset object
    """
    if mode == 'training':
        dataset = tf.data.Dataset.range(num_generators).apply(tf.data.experimental.parallel_interleave(
            training_dataset, cycle_length=num_generators, prefetch_input_elements=2*num_generators, sloppy=True))
    elif mode == 'validation':
        dataset = tf.data.Dataset.range(num_generators).apply(tf.data.experimental.parallel_interleave(
            validation_dataset, cycle_length=num_generators, prefetch_input_elements=2*num_generators, sloppy=True))
    return dataset.make_initializable_iterator()

def setup_optimizer():
    """ Sets up the optimizer to be used for gradient descent
    Returns:
        opt: The tf.optimizer object to use
        global_step: a TF Variable representing the total number of iterations on this model 
        """
    global training_sample_size
    # We initialize a dummy generator so we can get the training_sample_size
    dummy_gen = ClusterGenerator(FLAGS.train_data_root, mode='training', sessions_frac=FLAGS.dataset_fraction)
    training_sample_size = len(dummy_gen.train_clusters)

    if FLAGS.stepvalue is None:
        # With this stepvalue, the lr will decay by a factor of decay_per_10_epoch every 10 epochs
        decay_per_10_epoch = FLAGS.decay_per_10_epoch
        FLAGS.stepvalue = int(
            10 * np.log(FLAGS.gamma) * training_sample_size / np.log(decay_per_10_epoch))
    global_step = tf.Variable(0, trainable=False, name='global_step')
    lr_op = tf.train.exponential_decay(FLAGS.base_lr, global_step=global_step,
                                        decay_steps=FLAGS.stepvalue, decay_rate=FLAGS.gamma, name='lr')
    if FLAGS.optimizer == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(learning_rate=lr_op)
        return opt, global_step
    elif FLAGS.optimizer == 'momentum':
        opt = tf.train.MomentumOptimizer(
            learning_rate=lr_op, momentum=0.9, use_nesterov=False)
        return opt, global_step
    elif FLAGS.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate=lr_op)
        return opt, global_step
    else:
        logger.error("Optimizer {} is not implemented. Please use 'rmsprop' or 'momentum".format(
            FLAGS.optimizer))
        raise NotImplementedError

def initialize_trainer():
    """ Prints out some info and returns a validation summary file """
    train_session_start = time.time()
    logger.info("Training starting at time: {}".format(train_session_start))
    logger.info("Tensorflow version: {}".format(tf.__version__))
    logger.info("Flags: {}".format(FLAGS))
    mu.initialize_wandb(FLAGS)

    # Prepare validation summary 
    val_sum_file = os.path.join(
        FLAGS.logs_dir, 'validation_summary-{}.txt'.format(train_session_start))
    with file_io.FileIO(val_sum_file, 'w+') as f:
        header = 'train_step,val_loss,val_less_one,val_less_three\n'
        f.write(header)
    return val_sum_file

def get_batch(training_iterator, validation_iterator):
    """ Gets a batch of data for training or validation, and reshapes for input to network 
    Returns:
        images: Input images at full resolution
        cams: Numpy array encoding camera intrinsics and extrinsics. Scaled by FLAGS.sample_scale for use in homographies
        depth: Depth image for loss. Downsampled by FLAGS.sample_scale
        depth_start: The starting depth
        depth_interval: The distance between depth buckets
        full_depth: The depth image at full resolution
        """
    if training_status:
        images, cams, depth, full_depth = training_iterator.get_next()
    else:
        images, cams, depth, full_depth = validation_iterator.get_next()

    images.set_shape(tf.TensorShape(
        [None, FLAGS.view_num, None, None, 3]))
    cams.set_shape(tf.TensorShape(
        [None, FLAGS.view_num, 2, 4, 4]))
    depth.set_shape(
        tf.TensorShape([None, None, None, 1]))
    depth_start = tf.reshape(
        tf.slice(cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    depth_interval = tf.reshape(
        tf.slice(cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    depth_end = tf.reshape(
        tf.slice(cams, [0, 0, 1, 3, 3], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    return images, cams, depth, depth_start, depth_interval, full_depth, depth_end

def get_loss(images, cams, depth_image, depth_start, depth_interval, full_depth, depth_end, i):
    """ Performs inference with specified network and return loss function """
    is_master_gpu = True if i == 0 else False
    # inference
    if FLAGS.regularization == '3DCNNs':
        main_trainable = False if FLAGS.refinement_train_mode == 'refine_only' and FLAGS.refinement==True else True
        # initial depth map
        depth_map, prob_map = inference(
            images, cams, FLAGS.max_d, depth_start, depth_interval, FLAGS.network_mode, is_master_gpu, trainable=main_trainable, inverse_depth = FLAGS.inverse_depth)
        # refinement
        if FLAGS.refinement:
            #refine_trainable = False if FLAGS.refinement_train_mode == 'main_only' else True
            refine_trainable = True
            ref_image = tf.squeeze(
                tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
            stereo_image = None
            if images.shape[1] > 1 and FLAGS.refine_with_stereo:
                stereo_image = tf.squeeze(
                    tf.slice(images, [0, 1, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)

            refined_depth_map, residual_depth_map = depth_refine(depth_map, ref_image, prob_map, FLAGS.max_d, depth_start, depth_interval, FLAGS.network_mode, \
                FLAGS.refinement_network, is_master_gpu, trainable=refine_trainable, upsample_depth=FLAGS.upsample_before_refinement, refine_with_confidence=FLAGS.refine_with_confidence, stereo_image=stereo_image)
                                    # regression loss
            loss0, less_one_main, less_three_main = mvsnet_regression_loss(
                depth_map, depth_image, depth_start, depth_end)
            # If we upsampled the depth image to full resolution we need to compute loss with full_depth
            if FLAGS.upsample_before_refinement:
                loss1, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
                    refined_depth_map, full_depth, depth_start, depth_end)
            else:
                loss1, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
                    refined_depth_map, depth_image, depth_start, depth_end)
            if FLAGS.refinement_train_mode == 'refine_only':
                # If we are only training the refinement network we are only computing gradients wrt the refinement network params
                # These gradients on l0 will be zero, so no need to include l0 in the loss
                loss = loss1 + 1e-9*loss0
            elif FLAGS.refinement_train_mode == 'main_only':
                loss = loss0 + 1e-9*loss1
                less_one_accuracy = less_one_main
                less_three_accuracy = less_three_main
            else:
                loss = (loss0 + loss1) / 2
        else:
            # regression loss
            loss, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
                depth_map, depth_image, depth_start, depth_end)
        return loss, less_one_accuracy, less_three_accuracy

    elif FLAGS.regularization == 'GRU':
        # probability volume
        prob_volume = inference_prob_recurrent(
            images, cams, FLAGS.max_d, depth_start, depth_interval, FLAGS.network_mode, is_master_gpu)

        # classification loss
        loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
            mvsnet_classification_loss(
                prob_volume, depth_image, FLAGS.max_d, depth_start, depth_interval)
        return loss, less_one_accuracy, less_three_accuracy

def save_model(sess, saver, total_step, step):
    """" save model periodically """
    if (total_step % FLAGS.snapshot == 0 or step == (training_sample_size - 1)):
        model_folder = os.path.join(
            FLAGS.model_dir, FLAGS.regularization)
        ckpt_path = os.path.join(model_folder, 'model.ckpt')
        print(Notify.INFO, 'Saving model to %s' %
                ckpt_path, Notify.ENDC)
        saver.save(sess, ckpt_path, global_step=total_step)

def validate(sess, val_sum_file, loss, less_one_accuracy, less_three_accuracy, epoch, total_step):
    training_status = False
    val_loss = []
    val_less_one = []
    val_less_three = []
    for i in range(int(FLAGS.val_batch_size / FLAGS.num_gpus)):
        # run one batch
        start_time = time.time()
        try:
            out_loss, out_less_one, out_less_three = sess.run(
                [loss, less_one_accuracy, less_three_accuracy])
        except tf.errors.OutOfRangeError:
            # ==> "End of dataset"
            logger.warn("End of validation dataset")
            break
        duration = time.time() - start_time

        print(Notify.INFO, '_validating_',
                'epoch, %d, train step %d, val loss = %.4f, val (< 1px) = %.4f, val (< 3px) = %.4f (%.3f sec/step)' %
                (epoch, total_step, out_loss, out_less_one, out_less_three, duration), Notify.ENDC)

        val_loss.append(out_loss)
        val_less_one.append(out_less_one)
        val_less_three.append(out_less_three)
    l = np.mean(np.asarray(val_loss))
    l1 = np.mean(np.asarray(val_less_one))
    l3 = np.mean(np.asarray(val_less_three))

    print(Notify.INFO, '\n VAL STEP COMPLETED. Average loss: {}, Average less one: {}, Average less three: {}\n'.format(
        l, l1, l3))
    wandb.log({'val_loss':l,'val_less_one':l1,'val_less_three':l3}, step=total_step)

    with file_io.FileIO(val_sum_file, 'a+') as f:
        f.write('{},{},{},{}\n'.format(
            total_step, l, l1, l3))
    print(
        Notify.INFO, 'Validation output summary saved to: {}'.format(val_sum_file))



def train():
    """ Executes the main training loop for multiple epochs """
    val_sum_file = initialize_trainer()

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        ########## data iterator #########
        training_iterator = parallel_iterator('training')
        validation_iterator = parallel_iterator('validation')
        opt, global_step = setup_optimizer()    

        global training_status
        training_status = True  # This is set to true when training, false when validating
        tower_grads = [] # to keep track of the gradients across all towers.
        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Model_tower%d' % i) as scope:
                    images, cams, depth, depth_start, depth_interval, full_depth, depth_end = get_batch(training_iterator, validation_iterator)
                    loss, less_one_accuracy, less_three_accuracy = get_loss(images, cams, depth, depth_start, depth_interval, full_depth, depth_end,  i)
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        train_opt = opt.apply_gradients(grads, global_step=global_step)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        init_op, config = mu.init_session()

        with tf.Session(config=config) as sess:
            # initialization
            sess.run(init_op)
            total_step = load_model(sess)

            # training several epochs
            num_iterations = int(np.ceil(float(FLAGS.epoch) / float(FLAGS.num_gpus)))
            for epoch in range(num_iterations):

                # training of one epoch
                step = 0
                sess.run(training_iterator.initializer)
                sess.run(validation_iterator.initializer)

                for i in range(training_sample_size):
                    training_status = True
                    # run one batch
                    start_time = time.time()
                    try:
                        out_opt, out_loss, out_less_one, out_less_three = sess.run(
                            [train_opt, loss, less_one_accuracy, less_three_accuracy])
                    except tf.errors.OutOfRangeError:
                        logger.warn("End of dataset")
                        break
                    duration = time.time() - start_time

                    # print info
                    if step % FLAGS.display == 0:
                        print(Notify.INFO,
                              'epoch, %d, step %d, total_step %d, loss = %.4f, (< 1px) = %.4f, (< 3px) = %.4f (%.3f sec/step)' %
                              (epoch, step, total_step, out_loss, out_less_one, out_less_three, duration), Notify.ENDC)
                        wandb.log({'loss':out_loss,'less_one':out_less_one,'less_three':out_less_three,'time_per_step':duration},step=total_step)

                    save_model(sess, saver, total_step, step)
                    step += FLAGS.batch_size * FLAGS.num_gpus
                    total_step += FLAGS.batch_size * FLAGS.num_gpus

                    # Validate model against validation set of data
                    if i % FLAGS.train_steps_per_val == 0:
                        validate(sess, val_sum_file, loss, less_one_accuracy, less_three_accuracy, epoch, total_step)


def main(argv=None):  # pylint: disable=unused-argument
    """ program entrance """
    train()


if __name__ == '__main__':
    logger.info('Training MVSNet with %d views' % FLAGS.view_num)
    tf.app.run()

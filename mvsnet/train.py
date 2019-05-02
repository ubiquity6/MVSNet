#!/usr/bin/env python
from __future__ import print_function

from homography_warping import get_homographies, homography_warping
from model import *
from preprocess import *
from loss import *
from cnn_wrapper.common import Notify
from mvs_data_generation.cluster_generator import ClusterGenerator
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
import tensorflow as tf
from tensorflow.python.lib.io import file_io

# params for datasets
tf.app.flags.DEFINE_string('train_data_root', None,
                           """Path to dtu dataset.""")
tf.app.flags.DEFINE_string('log_dir', None,
                           """Path to store the log.""")
tf.app.flags.DEFINE_string('model_dir', None,
                           """Path to save the model.""")
tf.app.flags.DEFINE_string('job-dir', None,
                           """Path to save job artifacts""")
tf.app.flags.DEFINE_boolean('train_dtu', True,
                            """Whether to train.""")
tf.app.flags.DEFINE_boolean('use_pretrain', True,
                            """Whether to train.""")
tf.app.flags.DEFINE_integer('ckpt_step', None,
                            """ckpt step.""")

# input parameters
tf.app.flags.DEFINE_integer('view_num', 4,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 192,
                            """Maximum depth step when training.""")
tf.app.flags.DEFINE_integer('max_w', 640,
                            """Maximum image width when training.""")
tf.app.flags.DEFINE_integer('max_h', 480,
                            """Maximum image height when training.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25,
                          """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_float('interval_scale', 1.0,
                          """Downsample scale for building cost volume.""")
tf.app.flags.DEFINE_float('base_image_size', 8,
                          """Base image size""")
# network architectures
tf.app.flags.DEFINE_string('regularization', '3DCNNs',
                           """Regularization method.""")
tf.app.flags.DEFINE_string('optimizer', 'rmsprop',
                           """Optimizer to use. One of 'momentum' or 'rmsprop' """)
tf.app.flags.DEFINE_boolean('refinement', False,
                            """Whether to apply depth map refinement for 3DCNNs""")

# training parameters
tf.app.flags.DEFINE_integer('num_gpus', None,
                            """Number of GPUs.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Training batch size.""")
tf.app.flags.DEFINE_integer('epoch', None,
                            """Training epoch number.""")
tf.app.flags.DEFINE_float('val_ratio', 0,
                          """Ratio of validation set when splitting dataset.""")
tf.app.flags.DEFINE_float('base_lr', 0.0025,
                          """Base learning rate.""")
tf.app.flags.DEFINE_integer('display', 1,
                            """Interval of loginfo display.""")
tf.app.flags.DEFINE_integer('stepvalue', None,
                            """Step interval to decay learning rate.""")
tf.app.flags.DEFINE_integer('snapshot', 5000,
                            """Step interval to save the model.""")
tf.app.flags.DEFINE_float('gamma', 0.9,
                          """Learning rate decay rate.""")
tf.app.flags.DEFINE_float('val_batch_size', 15,
                          """Number of images to run validation on when validation.""")
tf.app.flags.DEFINE_float('train_steps_per_val', 200,
                          """Number of samples to train on before running a round of validation.""")

FLAGS = tf.app.flags.FLAGS

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
    gen = ClusterGenerator(FLAGS.train_data_root, FLAGS.view_num, FLAGS.max_w, FLAGS.max_h,
                                FLAGS.max_d, FLAGS.interval_scale, FLAGS.base_image_size, mode=mode, flip_cams=flip_cams)
    if mode == 'training':
        training_sample_size = len(gen.train_clusters)
    return iter(gen)


def training_dataset(n):
    """ Returns a dataset over the Training data, initialized from a data generator
    """
    generator_data_type = (tf.float32, tf.float32, tf.float32)
    training_set = tf.data.Dataset.from_generator(
        lambda: generator(n, mode='training'), generator_data_type)
    training_set = training_set.batch(FLAGS.batch_size)
    training_set = training_set.prefetch(buffer_size=1)
    return training_set


def validation_dataset(n):
    """ Returns a dataset over the Validation data, initialized from a data generator
    """
    generator_data_type = (tf.float32, tf.float32, tf.float32)
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

def train(training_list=None, validation_list=None):
    """ training mvsnet """

    train_session_start = time.time()
    print("Training starting at time:", train_session_start)
    print("Tensorflow version:", tf.__version__)
    print("Flags:", FLAGS)

    # Prepare validation summary file
    val_sum_file = os.path.join(
        FLAGS.log_dir, 'validation_summary-{}.txt'.format(train_session_start))
    with file_io.FileIO(val_sum_file, 'w+') as f:
        header = 'train_step,val_loss,val_less_one,val_less_three\n'
        f.write(header)

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        ########## data iterator #########
        # training generators
        train_gen = ClusterGenerator(FLAGS.train_data_root, FLAGS.view_num, FLAGS.max_w, FLAGS.max_h,
                                     FLAGS.max_d, FLAGS.interval_scale, FLAGS.base_image_size, mode='training')
        training_sample_size = len(train_gen.train_clusters)

        if FLAGS.regularization == 'GRU':
            training_sample_size = training_sample_size * 2

        training_iterator = parallel_iterator('training')
        validation_iterator = parallel_iterator('validation')

        training_status = True  # Set to true when training, false when validating

        ########## optimization options ##########
        if FLAGS.stepvalue is None:
            # With this stepvalue, the lr will decay by 0.5 every 10 epochs
            decay_per_10_epoch = 0.5
            FLAGS.stepvalue = int(10 * np.log(FLAGS.gamma) * training_sample_size / np.log(decay_per_10_epoch)  )
        global_step = tf.Variable(0, trainable=False, name='global_step')
        lr_op = tf.train.exponential_decay(FLAGS.base_lr, global_step=global_step,
                                           decay_steps=FLAGS.stepvalue, decay_rate=FLAGS.gamma, name='lr')
        if FLAGS.optimizer == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(learning_rate=lr_op)
        elif FLAGS.optimizer == 'momentum':
            opt = tf.train.MomentumOptimizer(learning_rate=lr_op, momentum=0.9, use_nesterov=False)
        else:
            print("Optimizer {} is not implemented. Please use 'rmsprop' or 'momentum".format(FLAGS.optimizer))
            sys.exit(1)


        tower_grads = []
        for i in xrange(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Model_tower%d' % i) as scope:
                    # generate data
                    if training_status:
                        images, cams, depth_image = training_iterator.get_next()
                    else:
                        images, cams, depth_image = validation_iterator.get_next()

                    images.set_shape(tf.TensorShape(
                        [None, FLAGS.view_num, None, None, 3]))
                    cams.set_shape(tf.TensorShape(
                        [None, FLAGS.view_num, 2, 4, 4]))
                    depth_image.set_shape(
                        tf.TensorShape([None, None, None, 1]))
                    depth_start = tf.reshape(
                        tf.slice(cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
                    depth_interval = tf.reshape(
                        tf.slice(cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])

                    is_master_gpu = False
                    if i == 0:
                        is_master_gpu = True

                    # inference
                    if FLAGS.regularization == '3DCNNs':

                        # initial depth map
                        depth_map, prob_map = inference(
                            images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)

                        # refinement
                        if FLAGS.refinement:
                            ref_image = tf.squeeze(
                                tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
                            refined_depth_map = depth_refine(depth_map, ref_image,
                                                             FLAGS.max_d, depth_start, depth_interval, is_master_gpu)
                                                    # regression loss
                            loss0, less_one_temp, less_three_temp = mvsnet_regression_loss(
                                depth_map, depth_image, depth_interval)
                            loss1, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
                                refined_depth_map, depth_image, depth_interval)
                            loss = (loss0 + loss1) / 2
                        else:
                            # regression loss
                            loss, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(
                                depth_map, depth_image, depth_interval)

                    elif FLAGS.regularization == 'GRU':

                        # probability volume
                        prob_volume = inference_prob_recurrent(
                            images, cams, FLAGS.max_d, depth_start, depth_interval, is_master_gpu)

                        # classification loss
                        loss, mae, less_one_accuracy, less_three_accuracy, depth_map = \
                            mvsnet_classification_loss(
                                prob_volume, depth_image, FLAGS.max_d, depth_start, depth_interval)

                    # retain the summaries from the final tower.
                    summaries = tf.get_collection(
                        tf.GraphKeys.SUMMARIES, scope)

                    # calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(loss)

                    # keep track of the gradients across all towers.
                    tower_grads.append(grads)

        # average gradient
        grads = average_gradients(tower_grads)

        # training opt
        train_opt = opt.apply_gradients(grads, global_step=global_step)

        # summary
        """
        summaries.append(tf.summary.scalar('loss', loss))
        summaries.append(tf.summary.scalar(
            'less_one_accuracy', less_one_accuracy))
        summaries.append(tf.summary.scalar(
            'less_three_accuracy', less_three_accuracy))
        summaries.append(tf.summary.scalar('lr', lr_op))
        weights_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in weights_list:
            summaries.append(tf.summary.histogram(var.op.name, var))
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(
                    var.op.name + '/gradients', grad))
                    """

        # saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        # summary_op = tf.summary.merge(summaries)

        # initialization option
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.inter_op_parallelism_threads = 0;
        config.intra_op_parallelism_threads = 0

        with tf.Session(config=config) as sess:

            # initialization
            total_step = 0
            sess.run(init_op)
            # summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

            # load pre-trained model
            if FLAGS.ckpt_step:
                pretrained_model_path = os.path.join(
                    FLAGS.model_dir, FLAGS.regularization, 'model.ckpt')
                restorer = tf.train.Saver(tf.global_variables())
                restorer.restore(
                    sess, '-'.join([pretrained_model_path, str(FLAGS.ckpt_step)]))
                print(Notify.INFO, 'Pre-trained model restored from %s' %
                      ('-'.join([pretrained_model_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
                total_step = FLAGS.ckpt_step

            # training several epochs
            # Because we have num_gpus parallelized generators, we actually go through 
            # the dataset num_gpus times in each loop below
            num_iterations = int(np.ceil(float(FLAGS.epoch) / float(FLAGS.num_gpus)))

            for epoch in range(num_iterations):

                # training of one epoch
                step = 0
                sess.run(training_iterator.initializer)
                sess.run(validation_iterator.initializer)

                # for i in range(int(training_sample_size / FLAGS.num_gpus)):
                for i in range(training_sample_size):
                    training_status = True
                    # run one batch
                    start_time = time.time()
                    try:
                        # out_summary_op, out_opt, out_loss, out_less_one, out_less_three = sess.run(
                        #    [summary_op, train_opt, loss, less_one_accuracy, less_three_accuracy])

                        out_opt, out_loss, out_less_one, out_less_three = sess.run(
                            [train_opt, loss, less_one_accuracy, less_three_accuracy])
                    except tf.errors.OutOfRangeError:
                        print("End of dataset")  # ==> "End of dataset"
                        break
                    duration = time.time() - start_time

                    # print info
                    if step % FLAGS.display == 0:
                        print(Notify.INFO,
                              'epoch, %d, step %d, total_step %d, loss = %.4f, (< 1px) = %.4f, (< 3px) = %.4f (%.3f sec/step)' %
                              (epoch, step, total_step, out_loss, out_less_one, out_less_three, duration), Notify.ENDC)

                    # write summary
                   # if step % (FLAGS.display * 10) == 0:
                   #     summary_writer.add_summary(out_summary_op, total_step)

                    # save the model checkpoint periodically
                    if (total_step % FLAGS.snapshot == 0 or step == (training_sample_size - 1)):
                        model_folder = os.path.join(
                            FLAGS.model_dir, FLAGS.regularization)
                       # if not os.path.exists(model_folder):
                        #    os.mkdir(model_folder)
                        ckpt_path = os.path.join(model_folder, 'model.ckpt')
                        print(Notify.INFO, 'Saving model to %s' %
                              ckpt_path, Notify.ENDC)
                        saver.save(sess, ckpt_path, global_step=total_step)
                    step += FLAGS.batch_size * FLAGS.num_gpus
                    total_step += FLAGS.batch_size * FLAGS.num_gpus

                    # Validate model against validation set of data
                    if i % FLAGS.train_steps_per_val == 0:
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
                                print("End of validation dataset")
                                break
                            duration = time.time() - start_time

                            # print info
                            if step % FLAGS.display == 0:
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

                        with file_io.FileIO(val_sum_file, 'a+') as f:
                            f.write('{},{},{},{}\n'.format(
                                total_step, l, l1, l3))
                        print(
                            Notify.INFO, 'Validation output summary saved to: {}'.format(val_sum_file))


def main(argv=None):  # pylint: disable=unused-argument
    """ program entrance """
    train()


if __name__ == '__main__':
    print('Training MVSNet with %d views' % FLAGS.view_num)
    tf.app.run()

#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Model architectures.
"""

import sys
import math
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS


def original_loss(y_true, y_pred, interval):
    """ non zero mean absolute loss for one batch
        This is the original loss function used in mvsnet paper"""
    with tf.name_scope('MAE'):
        shape = tf.shape(y_pred)
        interval = tf.reshape(interval, [shape[0]])
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.abs(tf.reduce_sum(mask_true, axis=[1, 2, 3])) + 1e-6
        masked_abs_error = tf.abs(
            mask_true * (y_true - y_pred))            # 4D
        masked_mae = tf.reduce_sum(
            masked_abs_error, axis=[1, 2, 3])        # 1D
        masked_mae = tf.reduce_sum((masked_mae / interval) / denom)         # 1
    return masked_mae, tf.no_op()


def power_loss(y_true, y_pred, interval, alpha, beta, no_interval_norm=False):
    """ non zero mean absolute loss for one batch

    This function parameterizes a loss of the general form:

    Loss = N * (|y_true-y_pred| + epsilon(y_true))^alpha / y_true^beta

    where alpha and beta are scalars, and N is a normalization constant which depends on
    alpha, beta and y_true. epsilon(y_true) is the expected noise of the measurement of y_true, and helps to prevent overfitting to noise
    in the depth map. This is the loss for an individual pixel. The total loss for a depth map
    is the average taken over all valid pixels.
    Additionally the numerator and denominator are multipled by a mask to mask out
    invalid pixels in the label. This was omitted above for notational simplicity.

    See this paper for a description and analysis of the noise model of the kinect sensor
    -- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3304120/

    One key takeaway is that the random error in Kinect depth maps increases quadratically with distance and
    reaches a maximum of 4cm at the maximum range of 5 meters
     """

    with tf.name_scope('MAE'):
        shape = tf.shape(y_pred)
        interval = tf.reshape(interval, [shape[0]])
        # mask_true is a tensor of 0's and 1's, where 1 is for valid pixels and 0 for invalid pixels
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        # The number of valid pixels in the depth map -- used for taking the average over only valid pixels
        num_valid_pixels = tf.abs(tf.reduce_sum(
            mask_true, axis=[1, 2, 3])) + 1e-6
        # Error is scaled by the depth itself
        denominator = y_true
        if beta == 0.0:
            # We treat the beta==0 case separately because 0^0 = 1
            denominator = num_valid_pixels
        else:
            denominator = tf.math.pow(y_true + 1e-9, beta)
            denominator = denominator*num_valid_pixels
        # Below we assume the random error in y_true used to regularize the divergence
        # increases linearly with distance
        epsilon = .005 * y_true
        numerator = tf.abs(y_true - y_pred) + epsilon
        if alpha != 1.0:
            numerator = tf.math.pow(
                numerator, alpha)
        # Apply the mask to the predicrions and labels
        numerator = numerator*mask_true
        # Divide the error by the distance
        loss = tf.reduce_sum(numerator / denominator, axis=[1, 2, 3])
        # The normalization is chosen so that, on average, the loss has approximately the same
        # magnitude as the original loss, regardless of the exponents chosen
        mean_true_depth = tf.reduce_sum(
            y_true * mask_true) / num_valid_pixels
        if no_interval_norm:
            normalization = tf.math.pow(
                mean_true_depth, beta)
        else:
            normalization = 10.0 * tf.math.pow(
                mean_true_depth, beta) / tf.math.pow(interval, alpha)
        loss = loss * normalization
    return loss, num_valid_pixels


def gaussian_loss(y_true, y_pred, interval, eta):
    """ non zero mean absolute loss for one batch

    This function parameterizes a loss of the form

    Loss = - exp(- x ^ 2 / 2*sigma ^ 2)

    where x = y_true - y_pred and
    sigma = eta * y_true

    and eta is a constant, generally much less than 1

    Args:
        y_true: true depth
        y_pred: predicted depth
        interval: depth interval used
        eta: multiplictive constant appearing in standard deviations of gaussian loss
    """

    with tf.name_scope('MAE'):
        shape = tf.shape(y_pred)
        interval = tf.reshape(interval, [shape[0]])
        # mask_true is a tensor of 0's and 1's, where 1 is for valid pixels and 0 for invalid pixels
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        # The number of valid pixels in the depth map -- used for taking the average over only valid pixels
        num_valid_pixels = tf.abs(tf.reduce_sum(
            mask_true, axis=[1, 2, 3])) + 1e-6
        # The standard deviation used in the gaussian is of the form eta * y_true
        # with a small offset to prevent division by zero on invalid pixels
        sigma = eta * y_true + 1e-6
        # Below we assume the random error in y_true used to regularize the divergence
        # increases linearly with distance
        error = y_true - y_pred
        error = error*mask_true
        x = - tf.math.pow(error / sigma, 2.0) / 2.0
        loss = - tf.math.exp(x)
        # Average over the number of valid pixels
        loss = tf.reduce_sum(loss) / num_valid_pixels
    return loss, tf.no_op()


def gradient_loss(y_true, y_pred, log=True):
    """ This loss term calculate the difference in depth gradients in the horizontal and vertical
    direction and returns the average absolution value of these gradients.
        Args:
            log: Whether or not to take the log of the depth gradients """
    with tf.name_scope('grad_loss'):
        mask = tf.cast(tf.not_equal(y_true, 0.0), dtype=tf.float32)
        num_valid_pixels = tf.reduce_sum(mask)
        diff = y_true - y_pred

        v_gradient = diff[0:-2, :] - diff[2:, :]
        v_mask = tf.math.multiply(mask[0:-2, :], mask[2:, :])
        v_gradient = tf.math.abs(tf.math.multiply(v_gradient, v_mask))

        h_gradient = diff[:, 0:-2] - diff[:, 2:]
        h_mask = tf.math.multiply(mask[:, 0:-2], mask[:, 2:])
        h_gradient = tf.math.abs(tf.math.multiply(h_gradient, h_mask))

        if log:
            # We add 1.0 since log(1) = 0 and log(0) = -infinity
            v_gradient = tf.math.log(1.0 + v_gradient)
            h_gradient = tf.math.log(1.0 + h_gradient)

        grad_loss = tf.reduce_sum(v_gradient) + tf.reduce_sum(h_gradient)
        grad_loss = grad_loss / num_valid_pixels
    return grad_loss


def less_one_percentage(y_true, y_pred, interval):
    """ less one accuracy for one batch """
    with tf.name_scope('less_one_error'):
        shape = tf.shape(y_pred)
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.abs(tf.reduce_sum(mask_true)) + 1e-6
        interval_image = tf.tile(tf.reshape(interval, [shape[0], 1, 1, 1]), [
            1, shape[1], shape[2], 1])
        abs_diff_image = tf.abs(y_true - y_pred) / interval_image
        less_one_image = mask_true * \
            tf.cast(tf.less_equal(abs_diff_image, 1.0), dtype='float32')
    return tf.reduce_sum(less_one_image) / denom


def less_three_percentage(y_true, y_pred, interval):
    """ less three accuracy for one batch """
    with tf.name_scope('less_three_error'):
        shape = tf.shape(y_pred)
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype='float32')
        denom = tf.abs(tf.reduce_sum(mask_true)) + 1e-6
        interval_image = tf.tile(tf.reshape(interval, [shape[0], 1, 1, 1]), [
            1, shape[1], shape[2], 1])
        abs_diff_image = tf.abs(y_true - y_pred) / interval_image
        less_three_image = mask_true * \
            tf.cast(tf.less_equal(abs_diff_image, 3.0), dtype='float32')
    return tf.reduce_sum(less_three_image) / denom


def mvsnet_regression_loss(estimated_depth_image, depth_image, depth_start, depth_end, loss_type='original', alpha=1.0, beta=0.0, eta=0.02, grad_loss=True):
    """ compute loss and accuracy """
    # For loss and accuracy we use a depth_interval that is independent of the number of depth buckets
    # so we can easily compare results for various depth_num. We divide by 191 for historical reasons.
    depth_interval = tf.div(depth_end-depth_start, 191.0)
    if loss_type == 'original':
        loss, debug = original_loss(
            depth_image, estimated_depth_image, depth_interval)
    elif loss_type == 'power':
        loss, debug = power_loss(
            depth_image, estimated_depth_image, depth_interval, alpha, beta)
    elif loss_type == 'gaussian':
        loss, debug = gaussian_loss(
            depth_image, estimated_depth_image, depth_interval, eta)
    else:
        raise NotImplementedError

    if grad_loss:
        gamma = 1.0
        g_loss = gradient_loss(depth_image, estimated_depth_image)
        loss = loss + gamma * g_loss
        debug = g_loss

    # less one accuracy
    less_one_accuracy = less_one_percentage(
        depth_image, estimated_depth_image, depth_interval)
    # less three accuracy
    less_three_accuracy = less_three_percentage(
        depth_image, estimated_depth_image, depth_interval)

    return loss, less_one_accuracy, less_three_accuracy, debug


def mvsnet_classification_loss(prob_volume, gt_depth_image, depth_num, depth_start, depth_interval):
    """ compute loss and accuracy """

    # get depth mask
    mask_true = tf.cast(tf.not_equal(gt_depth_image, 0.0), dtype='float32')
    valid_pixel_num = tf.reduce_sum(mask_true, axis=[1, 2, 3]) + 1e-7
    # gt depth map -> gt index map
    shape = tf.shape(gt_depth_image)
    depth_end = depth_start + \
        (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    start_mat = tf.tile(tf.reshape(depth_start, [shape[0], 1, 1, 1]), [
        1, shape[1], shape[2], 1])

    interval_mat = tf.tile(tf.reshape(depth_interval, [shape[0], 1, 1, 1]), [
        1, shape[1], shape[2], 1])
    gt_index_image = tf.div(gt_depth_image - start_mat, interval_mat)
    gt_index_image = tf.multiply(mask_true, gt_index_image)
    gt_index_image = tf.cast(tf.round(gt_index_image), dtype='int32')
    # gt index map -> gt one hot volume (B x H x W x 1)
    gt_index_volume = tf.one_hot(gt_index_image, depth_num, axis=1)
    # cross entropy image (B x H x W x 1)
    cross_entropy_image = - \
        tf.reduce_sum(gt_index_volume * tf.log(prob_volume), axis=1)
    # masked cross entropy loss
    masked_cross_entropy_image = tf.multiply(mask_true, cross_entropy_image)
    masked_cross_entropy = tf.reduce_sum(
        masked_cross_entropy_image, axis=[1, 2, 3])
    masked_cross_entropy = tf.reduce_sum(
        masked_cross_entropy / valid_pixel_num)

    # winner-take-all depth map
    wta_index_map = tf.cast(tf.argmax(prob_volume, axis=1), dtype='float32')
    wta_depth_map = wta_index_map * interval_mat + start_mat

    # non zero mean absulote loss
    masked_mae = non_zero_mean_absolute_diff(
        gt_depth_image, wta_depth_map, tf.abs(depth_interval))
    # less one accuracy
    less_one_accuracy = less_one_percentage(
        gt_depth_image, wta_depth_map, tf.abs(depth_interval))
    # less three accuracy
    less_three_accuracy = less_three_percentage(
        gt_depth_image, wta_depth_map, tf.abs(depth_interval))

    return masked_cross_entropy, masked_mae, less_one_accuracy, less_three_accuracy, wta_depth_map

#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Loss formulations.
"""

import sys
import math
import tensorflow as tf
import numpy as np
from mvsnet.cnn_wrapper.mvsnetworks import *
from mvsnet.convgru import ConvGRUCell
from mvsnet.homography_warping import *
from mvsnet.utils import setup_logger
logger = setup_logger('mvsnet.cnn_wrapper.model')

FLAGS = tf.app.flags.FLAGS


def get_probability_map(cv_batch, depth_map_batch, depth_start_batch, depth_interval_batch, inverse_depth = False, num_buckets=4):
    """ Gets the probability maps for depth predictions, slice by slice if batch_size > 1. See get_probability_map_slice for more info """
    shape = tf.shape(depth_map_batch)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    depth_num = tf.shape(cv_batch)[1]
    prob_map_slices = []
    for i in range(FLAGS.batch_size):
        cv = cv_batch[i]
        depth_map = depth_map_batch[i]
        cv = tf.reshape(cv, [1, depth_num, height, width ])
        depth_map = tf.reshape(depth_map, [1, height, width, 1 ])
        depth_start = depth_start_batch[i]
        depth_interval = depth_interval_batch[i]
        prob_map = get_probability_map_slice(cv, depth_map, depth_start, depth_interval, inverse_depth, num_buckets)
        prob_map_slices.append(prob_map)
    
    prob_map = tf.concat(prob_map_slices, axis=0)
    return prob_map





def get_probability_map_slice(cv, depth_map, depth_start, depth_interval, inverse_depth = False, num_buckets=4):
    """ get probability map from cost volume 
    The probability map is computed by summing the probabilities of the four depth slices int he cost volume that are closest
    to the predicted depth ~ this is a simple measure of confidence that works well for downstream tasks like fusion.
    Args:
        cv: Cost volume
        depth_map: The depth map
        depth_start: The minimum depth
        depth_interval: The depth bucket size
        inverse_depth: True if depth buckets are sampled uniformly in inverse depth space
        num_buckets: Number of depth buckets of cost volume to sum to compute probability map -- we support 2 or 4
    """

    def _repeat_(x, num_repeats):
        """ repeat each element num_repeats times """
        x = tf.reshape(x, [-1])
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1, 1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    shape = tf.shape(depth_map)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    depth_num = tf.shape(cv)[1]
    # dynamic gpu params

    # byx coordinate, batched & flattened
    b_coordinates = tf.range(batch_size)
    y_coordinates = tf.range(height)
    x_coordinates = tf.range(width)
    b_coordinates, y_coordinates, x_coordinates = tf.meshgrid(
        b_coordinates, y_coordinates, x_coordinates)
    b_coordinates = _repeat_(b_coordinates, batch_size)
    y_coordinates = _repeat_(y_coordinates, batch_size)
    x_coordinates = _repeat_(x_coordinates, batch_size)

    if inverse_depth:
        depth_end = depth_start + \
            (tf.cast(depth_num, tf.float32) - 1) * depth_interval
        inv_depth_start = tf.reshape(tf.div(1.0, depth_start), [])
        inv_depth_end = tf.reshape(tf.div(1.0, depth_end), [])
        inv_depth = tf.lin_space(inv_depth_start, inv_depth_end, depth_num)
        depth_samples = tf.div(1.0, inv_depth)
        # Here we compute the depth bucket indices to be used for probability averaging
        # Since we are using inverse depth, we compute them in inverse depth space
        inv_depth_interval = tf.div(
            (inv_depth_start - inv_depth_end), tf.cast(depth_num, tf.float32) - 1.0)
        inv_depth_data = tf.div(1.0, depth_map)
        inv_depth_data = tf.div(inv_depth_data - inv_depth_end, inv_depth_interval)
        inv_depth_data = tf.reshape(inv_depth_data,[-1])
        # We need to linearly invert the index to get the correct index in depth space
        d_coordinates_left0 = depth_num - tf.cast(tf.ceil(inv_depth_data), 'int32') - 1
        d_coordinates_left0 = tf.clip_by_value(d_coordinates_left0,0, depth_num-1)
        d_coordinates1_right0 = depth_num - \
            tf.cast(tf.floor(inv_depth_data), 'int32') - 1
        d_coordinates1_right0 = tf.clip_by_value(
            d_coordinates1_right0, 0, depth_num-1)
        d_coordinates_left1 = tf.clip_by_value(
            d_coordinates_left0 - 1, 0, depth_num - 1)
        d_coordinates1_right1 = tf.clip_by_value(
            d_coordinates1_right0 + 1, 0, depth_num - 1)

    else:
        # d coordinate (floored and ceiled), batched & flattened
        d_coordinates = tf.reshape(
            (depth_map - depth_start) / depth_interval, [-1])
        d_coordinates_left0 = tf.clip_by_value(
            tf.cast(tf.floor(d_coordinates), 'int32'), 0, depth_num - 1)
        d_coordinates_left1 = tf.clip_by_value(
            d_coordinates_left0 - 1, 0, depth_num - 1)
        d_coordinates1_right0 = tf.clip_by_value(
            tf.cast(tf.ceil(d_coordinates), 'int32'), 0, depth_num - 1)
        d_coordinates1_right1 = tf.clip_by_value(
            d_coordinates1_right0 + 1, 0, depth_num - 1)

    # voxel coordinates
    voxel_coordinates_left0 = tf.stack(
        [b_coordinates, d_coordinates_left0, y_coordinates, x_coordinates], axis=1)
    voxel_coordinates_right0 = tf.stack(
        [b_coordinates, d_coordinates1_right0, y_coordinates, x_coordinates], axis=1)
    # get probability image by gathering and interpolation
    prob_map_left0 = tf.gather_nd(cv, voxel_coordinates_left0)
    prob_map_right0 = tf.gather_nd(cv, voxel_coordinates_right0)
    prob_map = prob_map_left0 + prob_map_right0 

    if num_buckets == 4:
        # If num_buckets = 4 then we also add the probability in another bucket to left and right
        voxel_coordinates_right1 = tf.stack(
            [b_coordinates, d_coordinates1_right1, y_coordinates, x_coordinates], axis=1)
        voxel_coordinates_left1 = tf.stack(
            [b_coordinates, d_coordinates_left1, y_coordinates, x_coordinates], axis=1)
        prob_map_left1 = tf.gather_nd(cv, voxel_coordinates_left1)
        prob_map_right1 = tf.gather_nd(cv, voxel_coordinates_right1)
        prob_map += prob_map_left1 + prob_map_right1
    
    prob_map = tf.reshape(prob_map, [batch_size, height, width, 1])

    return prob_map


def get_probability_map_batch(cv, depth_map, depth_start, depth_interval, inverse_depth = False, num_buckets=4):
    """ get probability map from cost volume. works with batch_size > 1
    The probability map is computed by summing the probabilities of the four depth slices int he cost volume that are closest
    to the predicted depth ~ this is a simple measure of confidence that works well for downstream tasks like fusion.
    Args:
        cv: Cost volume
        depth_map: The depth map
        depth_start: The minimum depth
        depth_interval: The depth bucket size
        inverse_depth: True if depth buckets are sampled uniformly in inverse depth space
        num_buckets: Number of depth buckets of cost volume to sum to compute probability map -- we support 2 or 4
    """
    def _repeat_(x, num_repeats):
        x = tf.reshape(x, [-1])
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1, 1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    shape = tf.shape(depth_map)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    depth_num = tf.shape(cv)[1]
    # dynamic gpu params

    # byx coordinate, batched & flattened
    b_coordinates = tf.range(batch_size)
    y_coordinates = tf.range(height)
    x_coordinates = tf.range(width)
    b_coordinates, y_coordinates, x_coordinates = tf.meshgrid(
        b_coordinates, y_coordinates, x_coordinates)
    b_coordinates = _repeat_(b_coordinates, 1)
    y_coordinates = _repeat_(y_coordinates, 1)
    x_coordinates = _repeat_(x_coordinates, 1)

    if inverse_depth:
        depth_end = depth_start + \
            (tf.cast(depth_num, tf.float32) - 1) * depth_interval
        inv_depth_start = tf.reshape(tf.div(1.0, depth_start), [])
        inv_depth_end = tf.reshape(tf.div(1.0, depth_end), [])
        inv_depth = tf.lin_space(inv_depth_start, inv_depth_end, depth_num)
        depth_samples = tf.div(1.0, inv_depth)
        # Here we compute the depth bucket indices to be used for probability averaging
        # Since we are using inverse depth, we compute them in inverse depth space
        inv_depth_interval = tf.div(
            (inv_depth_start - inv_depth_end), tf.cast(depth_num, tf.float32) - 1.0)
        inv_depth_data = tf.div(1.0, depth_map)
        inv_depth_data = tf.div(inv_depth_data - inv_depth_end, inv_depth_interval)
        inv_depth_data = tf.reshape(inv_depth_data,[-1])
        # We need to linearly invert the index to get the correct index in depth space
        d_coordinates_left0 = depth_num - tf.cast(tf.ceil(inv_depth_data), 'int32') - 1
        d_coordinates_left0 = tf.clip_by_value(d_coordinates_left0,0, depth_num-1)
        d_coordinates1_right0 = depth_num - \
            tf.cast(tf.floor(inv_depth_data), 'int32') - 1
        d_coordinates1_right0 = tf.clip_by_value(
            d_coordinates1_right0, 0, depth_num-1)
        d_coordinates_left1 = tf.clip_by_value(
            d_coordinates_left0 - 1, 0, depth_num - 1)
        d_coordinates1_right1 = tf.clip_by_value(
            d_coordinates1_right0 + 1, 0, depth_num - 1)

    else:
        # d coordinate (floored and ceiled), batched & flattened
        # We need to subtract the starting point and divide by the depth interval for the depth map
        # but in order to do so, we need to broadcast depth_start and depth_interval to tensors of the correct shape
        # to apply the mathematical operation
        inv_depth_interval = tf.div(1.0, depth_interval)
        inv_depth_interval = tf.linalg.diag(inv_depth_interval)
        start_tensor = tf.ones((batch_size, height, width, 1))
        depth_start_mat = tf.linalg.diag(depth_start)
        start_tensor = tf.linalg.tensordot(depth_start_mat, start_tensor, [[1],[0]])

        shifted_depth = depth_map - start_tensor
        shifted_depth = tf.linalg.tensordot(inv_depth_interval,shifted_depth, [[1],[0]])
        d_coordinates = tf.reshape(shifted_depth, [-1])
        d_coordinates_left0 = tf.clip_by_value(
            tf.cast(tf.floor(d_coordinates), 'int32'), 0, depth_num - 1)
        d_coordinates_left1 = tf.clip_by_value(
            d_coordinates_left0 - 1, 0, depth_num - 1)
        d_coordinates1_right0 = tf.clip_by_value(
            tf.cast(tf.ceil(d_coordinates), 'int32'), 0, depth_num - 1)
        d_coordinates1_right1 = tf.clip_by_value(
            d_coordinates1_right0 + 1, 0, depth_num - 1)

    # voxel coordinates
    voxel_coordinates_left0 = tf.stack(
        [b_coordinates, d_coordinates_left0, y_coordinates, x_coordinates], axis=1)
    voxel_coordinates_right0 = tf.stack(
        [b_coordinates, d_coordinates1_right0, y_coordinates, x_coordinates], axis=1)
    # get probability image by gathering and interpolation
    prob_map_left0 = tf.gather_nd(cv, voxel_coordinates_left0)
    prob_map_right0 = tf.gather_nd(cv, voxel_coordinates_right0)
    prob_map = prob_map_left0 + prob_map_right0 

    if num_buckets == 4:
        # If num_buckets = 4 then we also add the probability in another bucket to left and right
        voxel_coordinates_right1 = tf.stack(
            [b_coordinates, d_coordinates1_right1, y_coordinates, x_coordinates], axis=1)
        voxel_coordinates_left1 = tf.stack(
            [b_coordinates, d_coordinates_left1, y_coordinates, x_coordinates], axis=1)
        prob_map_left1 = tf.gather_nd(cv, voxel_coordinates_left1)
        prob_map_right1 = tf.gather_nd(cv, voxel_coordinates_right1)
        prob_map += prob_map_left1 + prob_map_right1
    
    prob_map = tf.reshape(prob_map, [batch_size, height, width, 1])

    return prob_map


def inference(images, cams, depth_num, depth_start, depth_interval, network_mode, is_master_gpu=True, trainable=True, inverse_depth=False):
    """ infer depth image from multi-view images and cameras """

    # dynamic gpu params
    depth_end = depth_start + \
        (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # reference image
    ref_image = tf.squeeze(
        tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(
        tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction
    reuse = not is_master_gpu
    ref_tower = UNetDS2GN({'data': ref_image},
                          trainable=trainable, mode=network_mode, reuse=reuse)
    view_towers = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(
            tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UNetDS2GN(
            {'data': view_image}, trainable=trainable, mode=network_mode, reuse=True)
        view_towers.append(view_tower)

    """
    cam_residuals = []
    for view in range(1, FLAGS.view_num): 
        view_cam = tf.squeeze(
            tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        view_image = tf.squeeze(
            tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        cam_residual = OdometryNet(view_image, ref_image, view_cam)
        cam_residuals.append(view_image, ref_image)  
    how should we encode the view_cam into the camera network? I could do an encoding similar 
    to the one used in the gqn, where you basically concatenate the images and have a few layers
    with convolution and downsampling which then outputs multiple channels that are the size of 
    a camera pose matrix, and you concatenate those layers with the view_cam pose estimate and 
    then have a few densely connected layers to regress a final pose residual. Another option is to rectify the images
    with the estimate and then regress the residual. If there is a clean way to do this then I think this is the best option. I might be able to do the 
    rectification using a homograph at a fixed depth, say 1.0 meters out.

    """

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(
            tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        if inverse_depth:
            homographies = get_homographies_inv_depth(ref_cam, view_cam, depth_num=depth_num,
                                                      depth_start=depth_start, depth_end=depth_end)
        else:
            homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                            depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)

    # build cost volume by differentialble homography
    with tf.name_scope('cost_volume_homography'):
        depth_costs = []
        for d in range(depth_num):
            # compute cost (variation metric)
            ave_feature = ref_tower.get_output()
            ave_feature2 = tf.square(ref_tower.get_output())
            for view in range(0, FLAGS.view_num - 1):
                homography = tf.slice(view_homographies[view], begin=[
                                      0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
				# warped_view_feature = homography_warping(view_towers[view].get_output(), homography)
                warped_view_feature = tf_transform_homography(
                    view_towers[view].get_output(), homography)
                ave_feature = ave_feature + warped_view_feature
                ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
            ave_feature = ave_feature / FLAGS.view_num
            ave_feature2 = ave_feature2 / FLAGS.view_num
            cost = ave_feature2 - tf.square(ave_feature)
            depth_costs.append(cost)
        cost_volume = tf.stack(depth_costs, axis=1)

    # filtered cost volume, size of (B, D, H, W, 1)
    filtered_cost_volume_tower = RegNetUS0(
        {'data': cost_volume}, trainable=trainable, mode=network_mode, reuse=reuse)
    filtered_cost_volume = tf.squeeze(
        filtered_cost_volume_tower.get_output(), axis=-1)

    # depth map by softArgmin
    with tf.name_scope('soft_arg_min'):
        # probability volume by soft max
        probability_volume = tf.nn.softmax(
            tf.scalar_mul(-1, filtered_cost_volume), axis=1, name='prob_volume')
        # depth image by soft argmin
        volume_shape = tf.shape(probability_volume)
        soft_2d = []
        for i in range(FLAGS.batch_size):
            if inverse_depth:
                inv_depth_start = tf.reshape(tf.div(1.0, depth_start[i]), [])
                inv_depth_end = tf.reshape(tf.div(1.0, depth_end[i]), [])
                inv_depth = tf.lin_space(
                    inv_depth_start, inv_depth_end, tf.cast(depth_num, tf.int32))
                soft_1d = tf.div(1.0, inv_depth)
            else:
                soft_1d = tf.linspace(
                    depth_start[i], depth_end[i], tf.cast(depth_num, tf.int32))
            soft_2d.append(soft_1d)
        soft_2d = tf.reshape(tf.stack(soft_2d, axis=0), [
                             volume_shape[0], volume_shape[1], 1, 1])
        soft_4d = tf.tile(soft_2d, [1, 1, volume_shape[2], volume_shape[3]])
        estimated_depth_map = tf.reduce_sum(
            soft_4d * probability_volume, axis=1)
        estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=3)

    # probability map
    prob_map = get_probability_map(
        probability_volume, estimated_depth_map, depth_start, depth_interval, inverse_depth=inverse_depth)

    return estimated_depth_map, prob_map#, filtered_depth_map, probability_volume

def inference_mem(images, cams, depth_num, depth_start, depth_interval, network_mode, is_master_gpu=True, training=True, trainable=True, inverse_depth=False):
    """ inference of depth image from multi-view images and cameras """

    # dynamic gpu params
    depth_end = depth_start + \
        (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    feature_c = 32
    feature_h = FLAGS.height / 4
    feature_w = FLAGS.width / 4

    # reference image
    ref_image = tf.squeeze(
        tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(
        tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction
    reuse = tf.app.flags.FLAGS.reuse_vars #not is_master_gpu
    ref_tower = UNetDS2GN({'data': ref_image}, trainable=trainable,
                          training=training, mode=network_mode, reuse=reuse)
    base_divisor = ref_tower.base_divisor
    feature_c /= base_divisor
    ref_feature = ref_tower.get_output()
    ref_feature2 = tf.square(ref_feature)

    view_features = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(
            tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UNetDS2GN({'data': view_image}, trainable=trainable,
                               training=training, mode=network_mode, reuse=True)
        view_features.append(view_tower.get_output())
    view_features = tf.stack(view_features, axis=0)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(
            tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        if inverse_depth:
            homographies = get_homographies_inv_depth(ref_cam, view_cam, depth_num=depth_num,
                                                      depth_start=depth_start, depth_end=depth_end)
        else:
            homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                            depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)
    view_homographies = tf.stack(view_homographies, axis=0)

    # build cost volume by differentialble homography
    with tf.name_scope('cost_volume_homography'):
        depth_costs = []

        # Costs are computed at each depth, and across each view
        for d in range(depth_num):
            # compute cost (standard deviation feature)
            # This looks like  a single pass algorithm for standard deviation calculation (CH)
            ave_feature = tf.Variable(tf.zeros(
                [FLAGS.batch_size, feature_h, feature_w, feature_c]),
                name='ave', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            ave_feature2 = tf.Variable(tf.zeros(
                [FLAGS.batch_size, feature_h, feature_w, feature_c]),
                name='ave2', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            ave_feature = tf.assign(ave_feature, ref_feature)
            ave_feature2 = tf.assign(ave_feature2, ref_feature2)

            def body(view, ave_feature, ave_feature2):
                """Loop body."""
                homography = tf.slice(view_homographies[view], begin=[
                                      0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                # warped_view_feature = homography_warping(view_features[view], homography)
                warped_view_feature = tf_transform_homography(
                    view_features[view], homography)
                ave_feature = tf.assign_add(ave_feature, warped_view_feature)
                ave_feature2 = tf.assign_add(
                    ave_feature2, tf.square(warped_view_feature))
                view = tf.add(view, 1)
                return view, ave_feature, ave_feature2

            view = tf.constant(0)
            cond = lambda view, *_: tf.less(view, FLAGS.view_num - 1)
            _, ave_feature, ave_feature2 = tf.while_loop(
                cond, body, [view, ave_feature, ave_feature2], back_prop=False, parallel_iterations=1)

            ave_feature = tf.assign(ave_feature, tf.square(
                ave_feature) / (FLAGS.view_num * FLAGS.view_num))
            ave_feature2 = tf.assign(
                ave_feature2, ave_feature2 / FLAGS.view_num - ave_feature)
            depth_costs.append(ave_feature2)
        cost_volume = tf.stack(depth_costs, axis=1)

    # filtered cost volume, size of (B, D, H, W, 1)
    filtered_cost_volume_tower = RegNetUS0(
        {'data': cost_volume}, trainable=trainable, training=training, mode=network_mode, reuse=reuse)
    filtered_cost_volume = tf.squeeze(
        filtered_cost_volume_tower.get_output(), axis=-1)

    # depth map by softArgmin
    with tf.name_scope('soft_arg_min'):
        # probability volume by soft max
        probability_volume = tf.nn.softmax(tf.scalar_mul(-1, filtered_cost_volume),
                                           axis=1, name='prob_volume')
        # depth image by soft argmin
        volume_shape = tf.shape(probability_volume)
        soft_2d = []
        for i in range(FLAGS.batch_size):
            if inverse_depth:
                inv_depth_start = tf.reshape(tf.div(1.0, depth_start[i]), [])
                inv_depth_end = tf.reshape(tf.div(1.0, depth_end[i]), [])
                inv_depth = tf.lin_space(
                    inv_depth_start, inv_depth_end, tf.cast(depth_num, tf.int32))
                soft_1d = tf.div(1.0, inv_depth)
            else:
                soft_1d = tf.linspace(
                    depth_start[i], depth_end[i], tf.cast(depth_num, tf.int32))
            soft_2d.append(soft_1d)
        soft_2d = tf.reshape(tf.stack(soft_2d, axis=0), [
                             volume_shape[0], volume_shape[1], 1, 1])
        soft_4d = tf.tile(soft_2d, [1, 1, volume_shape[2], volume_shape[3]])
        estimated_depth_map = tf.reduce_sum(
            soft_4d * probability_volume, axis=1)
        estimated_depth_map = tf.expand_dims(estimated_depth_map, axis=3)

    # probability map
    prob_map = get_probability_map(
        probability_volume, estimated_depth_map, depth_start, depth_interval, inverse_depth=inverse_depth)

    # return filtered_depth_map,
    return estimated_depth_map, prob_map


def inference_prob_recurrent(images, cams, depth_num, depth_start, depth_interval, network_mode, is_master_gpu=True, trainable=True):
    """ infer disparity image from stereo images and cameras """

    # dynamic gpu params
    depth_end = depth_start + \
        (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # reference image
    ref_image = tf.squeeze(
        tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(
        tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction
    reuse = not is_master_gpu
    ref_tower = UNetDS2GN({'data': ref_image},
                          trainable=trainable, mode=network_mode, reuse=reuse)

    view_towers = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(
            tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UNetDS2GN(
            {'data': view_image}, trainable=trainable, mode=network_mode, reuse=True)
        view_towers.append(view_tower)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(
            tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                        depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)

    base_divisor = 1 if network_mode == 'normal' else 2

    gru1_filters = int(16 / base_divisor)
    gru2_filters = int(4 / base_divisor)
    gru3_filters = int(2 / base_divisor)
    feature_shape = [FLAGS.batch_size, FLAGS.height/4, FLAGS.width/4, 32]
    gru_input_shape = [feature_shape[1], feature_shape[2]]
    state1 = tf.zeros([FLAGS.batch_size, feature_shape[1],
                      feature_shape[2], gru1_filters])
    state2 = tf.zeros([FLAGS.batch_size, feature_shape[1],
                      feature_shape[2], gru2_filters])
    state3 = tf.zeros([FLAGS.batch_size, feature_shape[1],
                      feature_shape[2], gru3_filters])
    conv_gru1 = ConvGRUCell(shape=gru_input_shape, kernel=[
                            3, 3], filters=gru1_filters, trainable=trainable)
    conv_gru2 = ConvGRUCell(shape=gru_input_shape, kernel=[
                            3, 3], filters=gru2_filters, trainable=trainable)
    conv_gru3 = ConvGRUCell(shape=gru_input_shape, kernel=[
                            3, 3], filters=gru3_filters, trainable=trainable)

    exp_div = tf.zeros(
        [FLAGS.batch_size, feature_shape[1], feature_shape[2], 1])
    soft_depth_map = tf.zeros(
        [FLAGS.batch_size, feature_shape[1], feature_shape[2], 1])

    with tf.name_scope('cost_volume_homography'):

        # forward cost volume
        depth_costs = []
        for d in range(depth_num):

            # compute cost (variation metric)
            ave_feature = ref_tower.get_output()
            ave_feature2 = tf.square(ref_tower.get_output())

            for view in range(0, FLAGS.view_num - 1):
                homography = tf.slice(
                    view_homographies[view], begin=[0, d, 0, 0], size=[-1, 1, 3, 3])
                homography = tf.squeeze(homography, axis=1)
                # warped_view_feature = homography_warping(view_towers[view].get_output(), homography)
                warped_view_feature = tf_transform_homography(
                    view_towers[view].get_output(), homography)
                ave_feature = ave_feature + warped_view_feature
                ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
            ave_feature = ave_feature / FLAGS.view_num
            ave_feature2 = ave_feature2 / FLAGS.view_num
            cost = ave_feature2 - tf.square(ave_feature)

            # gru
            reg_cost1, state1 = conv_gru1(-cost, state1, scope='conv_gru1')
            reg_cost2, state2 = conv_gru2(reg_cost1, state2, scope='conv_gru2')
            reg_cost3, state3 = conv_gru3(reg_cost2, state3, scope='conv_gru3')
            reg_cost = tf.layers.conv2d(
                reg_cost3, 1, 3, padding='same', reuse=tf.AUTO_REUSE, name='prob_conv')
            depth_costs.append(reg_cost)

        prob_volume = tf.stack(depth_costs, axis=1)
        prob_volume = tf.nn.softmax(prob_volume, axis=1, name='prob_volume')

    return prob_volume

def inference_winner_take_all(images, cams, depth_num, depth_start, depth_end, network_mode,
                              is_master_gpu=True, reg_type='GRU', inverse_depth=False, training=True, trainable=True):
    """ infer disparity image from stereo images and cameras """

    if not inverse_depth:
        depth_interval = (depth_end - depth_start) / \
                          (tf.cast(depth_num, tf.float32) - 1)

    # reference image
    ref_image = tf.squeeze(
        tf.slice(images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
    ref_cam = tf.squeeze(
        tf.slice(cams, [0, 0, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)

    # image feature extraction
    reuse = not is_master_gpu
    ref_tower = UNetDS2GN({'data': ref_image}, trainable=trainable,
                          training=training, mode=network_mode, reuse=reuse)
    view_towers = []
    for view in range(1, FLAGS.view_num):
        view_image = tf.squeeze(
            tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        view_tower = UNetDS2GN({'data': view_image}, trainable=trainable,
                               training=training, mode=network_mode, reuse=True)
        view_towers.append(view_tower)

    # get all homographies
    view_homographies = []
    for view in range(1, FLAGS.view_num):
        view_cam = tf.squeeze(
            tf.slice(cams, [0, view, 0, 0, 0], [-1, 1, 2, 4, 4]), axis=1)
        if inverse_depth:
            homographies = get_homographies_inv_depth(ref_cam, view_cam, depth_num=depth_num,
                                depth_start=depth_start, depth_end=depth_end)
        else:
            homographies = get_homographies(ref_cam, view_cam, depth_num=depth_num,
                                            depth_start=depth_start, depth_interval=depth_interval)
        view_homographies.append(homographies)

    # gru unit
    base_divisor = 1 if network_mode == 'normal' else 2

    gru1_filters = int(16 / base_divisor)
    gru2_filters = int(4 / base_divisor)
    gru3_filters = int(2 / base_divisor)

    feature_shape = [FLAGS.batch_size, FLAGS.height/4, FLAGS.width/4, 32]
    gru_input_shape = [feature_shape[1], feature_shape[2]]
    state1 = tf.zeros([FLAGS.batch_size, feature_shape[1],
                      feature_shape[2], gru1_filters])
    state2 = tf.zeros([FLAGS.batch_size, feature_shape[1],
                      feature_shape[2], gru2_filters])
    state3 = tf.zeros([FLAGS.batch_size, feature_shape[1],
                      feature_shape[2], gru3_filters])
    conv_gru1 = ConvGRUCell(shape=gru_input_shape, kernel=[
                            3, 3], filters=gru1_filters, trainable=trainable)
    conv_gru2 = ConvGRUCell(shape=gru_input_shape, kernel=[
                            3, 3], filters=gru2_filters, trainable=trainable)
    conv_gru3 = ConvGRUCell(shape=gru_input_shape, kernel=[
                            3, 3], filters=gru3_filters, trainable=trainable)

    # initialize variables
    exp_sum = tf.Variable(tf.zeros(
        [FLAGS.batch_size, feature_shape[1], feature_shape[2], 1]),
        name='exp_sum', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    depth_image = tf.Variable(tf.zeros(
        [FLAGS.batch_size, feature_shape[1], feature_shape[2], 1]),
        name='depth_image', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    max_prob_image = tf.Variable(tf.zeros(
        [FLAGS.batch_size, feature_shape[1], feature_shape[2], 1]),
        name='max_prob_image', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    init_map = tf.zeros(
        [FLAGS.batch_size, feature_shape[1], feature_shape[2], 1])

    # define winner take all loop
    def body(depth_index, state1, state2, state3, depth_image, max_prob_image, exp_sum, incre):
        """Loop body."""

        # calculate cost
        ave_feature = ref_tower.get_output()
        ave_feature2 = tf.square(ref_tower.get_output())
        for view in range(0, FLAGS.view_num - 1):
            homographies = view_homographies[view]
            homographies = tf.transpose(homographies, perm=[1, 0, 2, 3])
            homography = homographies[depth_index]
            # warped_view_feature = homography_warping(view_towers[view].get_output(), homography)
            warped_view_feature = tf_transform_homography(
                view_towers[view].get_output(), homography)
            ave_feature = ave_feature + warped_view_feature
            ave_feature2 = ave_feature2 + tf.square(warped_view_feature)
        ave_feature = ave_feature / FLAGS.view_num
        ave_feature2 = ave_feature2 / FLAGS.view_num
        cost = ave_feature2 - tf.square(ave_feature)
        cost.set_shape(
            [FLAGS.batch_size, feature_shape[1], feature_shape[2], 32])

        # gru
        reg_cost1, state1 = conv_gru1(-cost, state1, scope='conv_gru1')
        reg_cost2, state2 = conv_gru2(reg_cost1, state2, scope='conv_gru2')
        reg_cost3, state3 = conv_gru3(reg_cost2, state3, scope='conv_gru3')
        reg_cost = tf.layers.conv2d(
            reg_cost3, 1, 3, padding='same', reuse=tf.AUTO_REUSE, name='prob_conv')
        prob = tf.exp(reg_cost)

        # index
        d_idx = tf.cast(depth_index, tf.float32)
        if inverse_depth:
            inv_depth_start = tf.div(1.0, depth_start)
            inv_depth_end = tf.div(1.0, depth_end)
            inv_interval = (inv_depth_start - inv_depth_end) / \
                            (tf.cast(depth_num, 'float32') - 1)
            inv_depth = inv_depth_start - d_idx * inv_interval
            depth = tf.div(1.0, inv_depth)
        else:
            depth = depth_start + d_idx * depth_interval
        temp_depth_image = tf.reshape(depth, [FLAGS.batch_size, 1, 1, 1])
        temp_depth_image = tf.tile(
            temp_depth_image, [1, feature_shape[1], feature_shape[2], 1])

        # update the best
        update_flag_image = tf.cast(
            tf.less(max_prob_image, prob), dtype='float32')
        new_max_prob_image = update_flag_image * prob + \
            (1 - update_flag_image) * max_prob_image
        new_depth_image = update_flag_image * temp_depth_image + \
            (1 - update_flag_image) * depth_image
        max_prob_image = tf.assign(max_prob_image, new_max_prob_image)
        depth_image = tf.assign(depth_image, new_depth_image)

        # update counter
        exp_sum = tf.assign_add(exp_sum, prob)
        depth_index = tf.add(depth_index, incre)

        return depth_index, state1, state2, state3, depth_image, max_prob_image, exp_sum, incre

    # run forward loop
    exp_sum = tf.assign(exp_sum, init_map)
    depth_image = tf.assign(depth_image, init_map)
    max_prob_image = tf.assign(max_prob_image, init_map)
    depth_index = tf.constant(0)
    incre = tf.constant(1)
    cond = lambda depth_index, *_: tf.less(depth_index, depth_num)
    _, state1, state2, state3, depth_image, max_prob_image, exp_sum, incre = tf.while_loop(
        cond, body
        , [depth_index, state1, state2, state3, depth_image, max_prob_image, exp_sum, incre]
        , back_prop=False, parallel_iterations=1)

    # get output
    forward_exp_sum = exp_sum + 1e-7
    forward_depth_map = depth_image
    return forward_depth_map, max_prob_image / forward_exp_sum

def depth_refine(init_depth_map, image, prob_map, depth_num, depth_start, depth_interval, network_mode, network_type, \
    is_master_gpu=True, training=True, trainable=True, upsample_depth=False, refine_with_confidence=False, stereo_image=None, residual_refinement=True):
    """ refine depth image with the image """

    # normalization parameters
    depth_shape = tf.shape(init_depth_map)
    image_shape = tf.shape(image)
    depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
    depth_scale = depth_end - depth_start

    # normalize depth map (to 0~1) so that refinement network learns a scale invariant refinement of the depth map
    init_norm_depth_map = tf.div(
        init_depth_map - depth_start, depth_scale)

    if upsample_depth:
        # Upsample depth map to resolution of input image
        init_norm_depth_map = tf.image.resize_bilinear(init_norm_depth_map, [image_shape[1], image_shape[2]])
        init_depth_map = tf.image.resize_bilinear(init_depth_map, [image_shape[1], image_shape[2]])
        if refine_with_confidence:
            # TODO: resize the probability map with nearest neighbor interpolation instead of bilinear
            prob_map = tf.image.resize_bilinear(prob_map, [image_shape[1], image_shape[2]])
    else:
        # Downsample original image to size of depth map
        image = tf.image.resize_bilinear(image, [depth_shape[1], depth_shape[2]])
        if stereo_image is not None:
            stereo_image = tf.image.resize_bilinear(
                stereo_image, [depth_shape[1], depth_shape[2]])
    
    data = init_norm_depth_map
    # Concatenates with confidence map
    if refine_with_confidence:
        data = tf.concat([data, prob_map], axis=3)

    # Concatenates with stereo partner
    if stereo_image is not None:
        data = tf.concat(
            [data, stereo_image], axis=3)
    # refinement network
    reuse = not is_master_gpu
    if tf.app.flags.FLAGS.reuse_vars:
        reuse = True
    if network_type == 'unet':
        norm_depth_tower = RefineUNetConv({'color_image': image, 'depth_image': data},
                                        trainable=trainable, training=training, mode=network_mode, reuse=reuse)
    elif network_type == 'original':
        norm_depth_tower = RefineNetConv({'color_image': image, 'depth_image': data},
                                        trainable=trainable, training=training, mode=network_mode, reuse=reuse)
    else:
        raise NotImplementedError

    residual_norm_depth_map = norm_depth_tower.get_output()
    residual_depth_map = tf.multiply(residual_norm_depth_map, depth_scale)
    # residual_refinement controls whether the refinement network predicts the residual or the depth map itself
    if residual_refinement:
        refined_depth_map = tf.add_n((residual_depth_map, init_depth_map), name='add_residual')
    else:
        refined_depth_map = residual_depth_map

    return refined_depth_map, residual_depth_map

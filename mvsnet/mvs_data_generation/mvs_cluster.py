import numpy as np
from utils import mask_depth_image, scale_camera, scale_image, center_image, set_log_level
import imageio
import logging
import json
import os
import scipy
import cv2
from tensorflow.python.lib.io import file_io
import tensorflow as tf

# Flag that determines if we are running on GCP or local
if 'CLOUD_ML_JOB_ID' in os.environ:
    GCP = True
else:
    GCP = False
if GCP:
    tf.enable_eager_execution()
"""

Cluster objects represent the visibility information used for MVS reconstruction

"""


"""
Copyright 2019, Chris Heinrich, Ubiquity6.
"""


class Cluster:
    def __init__(self, session_dir, ref_index, views, min_depth, max_depth, view_num,
                 image_width=1024, image_height=768, depth_num=256, interval_scale=1.0):
        logging.basicConfig()
        self.logger = logging.getLogger('Cluster')
        set_log_level(self.logger)
        self.session_dir = session_dir
        self.ref_index = int(ref_index)
        self.views = views
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.view_num = view_num
        self.image_width = image_width
        self.image_height = image_height
        self.depth_num = depth_num  # The number of depth buckets
        # An optional rescaling along the depth axis
        self.interval_scale = interval_scale
        self.set_indices()
        self.rescale = 1.0

    def image_path(self, index):
        return os.path.join(self.session_dir, 'images', '{}.jpg'.format(index))

    def depth_path(self, index):
        return os.path.join(self.session_dir, 'depths', '{}.png'.format(index))

    def camera_path(self, index):
        return os.path.join(self.session_dir, 'cameras', '{}.json'.format(index))

    def load_image(self, index):
        image_file = file_io.FileIO(self.image_path(index), mode='r')
        image = scipy.misc.imread(image_file, mode='RGB')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def load_depth(self, index):
        try:
            if GCP:
                depth_raw = tf.read_file(self.depth_path(index))
                depth = tf.image.decode_png(depth_raw, dtype=tf.uint16).numpy()
                return depth
            else:
                return imageio.imread(self.depth_path(index)).astype(np.uint16)
        except Exception as e:
            self.logger.warn('Depth map at path {} does not exist'.format(
                self.depth_path(index)))
            return None

    def load_camera(self, index):
        """ Loads all camera data into a single numpy array 
        of shape (2,4,4). The 0'th 4x4 slice is the 4x4 pose matrix
        and the 1'th 4x4 slice contains the intrinsics as well as the 
        depth min and depth max.
        """
        with file_io.FileIO(self.camera_path(index), mode='r') as c:
            camera_data = json.load(c)

        depth_interval = ((self.max_depth - self.min_depth) /
                          self.depth_num) * self.interval_scale

        cam = np.zeros((2, 4, 4))
        cam[0] = self.pose_matrix(camera_data)
        cam[0, 0:3, 3] *= 1000  # convert translation vector from meters to mmm
        cam[1, 0:3, 0:3] = self.intrinsics_matrix(camera_data)
        # This uses the convention from the original MVSNet paper where
        # depth information is stored in the bottom row of camera matrix
        cam[1, 3, 0], cam[1, 3, 1] = self.min_depth, depth_interval
        cam[1, 3, 2], cam[1, 3, 3] = self.depth_num, self.max_depth
        return cam

    def intrinsics_matrix(self, camera_data, focal_rescale=1.0):
        # dtu magic focal_rescale = 1.171875
        if 'dtu_scan_' in self.session_dir:
            focal_rescale = 1.171875
        mat = np.zeros((3, 3))
        intrin = camera_data["intrinsics"]
        mat[0, 0], mat[1, 1] = intrin["fx"] * \
            focal_rescale, intrin["fy"]*focal_rescale
        mat[0, 2], mat[1, 2], mat[2, 2] = intrin["px"], intrin["py"], 1.0
        return mat

    def pose_matrix(self, camera_data):
        mat = np.zeros((4, 4))
        data = camera_data["pose"]["matrix"]
        for i in range(4):
            for j in range(4):
                mat[i, j] = data["{},{}".format(i, j)]
        return mat

    def set_indices(self):
        indices = []
        indices.append(int(self.ref_index))  # ref_index should be set first
        for view in self.views:
            indices.append(int(view))
        # The number of indices should be exactly equal to view_num. If we don't have enough covisible views
        # then we add additional copies of the reference image to the list of indices
        diff = self.view_num - len(indices)
        if diff > 0:
            for i in range(diff):
                indices.append(int(self.ref_index))
        self.indices = indices[:self.view_num]

    def cameras(self):
        cams = []
        for index in self.indices:
            cams.append(self.load_camera(index))
        return cams

    def images(self, centered=True):
        images = []
        for index in self.indices:
            images.append(self.load_image(index))
        self.set_rescale(images)
        if len(images) > 0:
            self.original_image_shape = images[0].shape
        return images

    def depth_maps(self):
        depth_maps = []
        for index in self.indices:
            depth_maps.append(self.load_depth(index))
        return depth_maps

    def reference_depth(self):
        return self.load_depth(self.ref_index)

    def masked_reference_depth(self):
        depth = self.reference_depth()
        # Make sure depth has same scale as reference image
        try:
            scale = float(self.original_image_shape[0]) / float(depth.shape[0])
            depth = scale_image(
                depth, scale=scale, interpolation='nearest')
        except Exception as e:
            self.logger.warn('Failed to resize depth to input image size')
            pass

        return mask_depth_image(depth, self.min_depth, self.max_depth)

    def set_rescale(self, images):
        h_scale = 0
        w_scale = 0
        for view in range(len(images)):
            height_scale = float(self.image_height) / \
                images[view].shape[0]
            width_scale = float(self.image_width) / \
                images[view].shape[1]
            if height_scale > h_scale:
                h_scale = height_scale
            if width_scale > w_scale:
                w_scale = width_scale
        self.rescale = max(h_scale, w_scale)
        return self.rescale

    def scale_images(self, images):
        for i in range(len(images)):
            images[i] = scale_image(images[i], self.rescale)
        return images

    def scale_cameras(self, cams):
        for i in range(len(cams)):
            cams[i] = scale_camera(cams[i], self.rescale)
        return cams

    def center_images(self, images):
        for i in range(len(images)):
            images[i] = center_image(images[i])
        return images

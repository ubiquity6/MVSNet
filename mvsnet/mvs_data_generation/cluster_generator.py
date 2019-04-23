import os
from mvs_cluster import Cluster
import utils as ut
from random import Random
import numpy as np
import imageio
import time
import json
import logging
import tensorflow as tf
from tensorflow.python.lib.io import file_io
logging.basicConfig()

"""
Copyright 2019, Chris Heinrich, Ubiquity6.
"""


class ClusterGenerator:
    def __init__(self, sessions_dir, view_num, image_width=1024, image_height=768, depth_num=256,
                 interval_scale=1, base_image_size=1, include_empty=False, mode='training', val_split=0.1, rescaling=True, output_scale=0.25, flip_cams=True):
        # Setup logger

        self.logger = logging.getLogger('ClusterGenerator')
        ut.set_log_level(self.logger)
        self.sessions_dir = sessions_dir
        self.view_num = view_num
        self.image_width = image_width
        self.image_height = image_height
        self.depth_num = depth_num  # The number of depth buckets
        # An optional rescaling along the depth axis
        self.interval_scale = interval_scale
        # This is the number that image width and height need to be divisible by in order
        # for data to fit into network cleanly.
        self.base_image_size = base_image_size
        # Whether or not to include clusters w/ zero covisible views
        self.include_empty = include_empty
        self.mode = mode  # One of 'training' or 'validation'
        self.val_split = val_split  # Fraction of clusters to use for validation
        self.rescaling = rescaling
        # Factor by which output is scaled relative to input
        self.output_scale = output_scale
        self.flip_cams = flip_cams
        self.parse_sessions()
        self.set_iter_clusters()

    def parse_sessions(self):
        """ 
        Parses a directory of mvs training sessions and returns a 
        list of dictionaries describing visibility clusters from all sessions. If running
        in 'training' or 'validation' mode then self.sessions_dir is expected to include multiple subdirectories
        with individual sessions. If running in 'test' mode then self.sessions_dir is expected to include a single
        session, which is what will be used for computing depth  maps.

        Returns:
            clusters: A list of Cluster objects. See mvs_cluster.py for their declaration.
        """

        clusters = []
        if self.mode == 'test':
            self.load_clusters(self.sessions_dir, clusters)
        else:
            sessions = [f for f in tf.gfile.ListDirectory(
                self.sessions_dir) if not f.startswith('.') if not f.endswith('.txt')]
            for session in sessions:
                session_dir = os.path.join(self.sessions_dir, session)
                self.load_clusters(session_dir, clusters)

        self.logger.info(" There are {} clusters".format(len(clusters)))
        self.clusters = clusters
        return clusters

    def load_clusters(self, session_dir, clusters):
        with file_io.FileIO(os.path.join(session_dir, 'covisibility.json'), mode='r') as f:
            data = json.load(f)
        for d in data:
            if not self.include_empty and not data[d]['views']:
                # Skip if there are no covisible views and we don't include empty
                pass
            else:
                cluster = Cluster(session_dir, int(d), data[d]['views'], data[d]['min_depth'],
                                  data[d]['max_depth'], self.view_num, self.image_width, self.image_height, self.depth_num, self.interval_scale)
                clusters.append(cluster)

    def get_clusters(self):
        """ Gets mvs clusters for training and validation. It shuffles the clusters
        from their original order, but does so deterministically.

        Args:
            sessions_dir: The location of mvs-training sessions
            include_empty: Whether or not to include clusters w/o covisible views
            val_split: The fraction of clusters to use for training
        Returns:
            train_clusters: A list of clusters to use for training
            val_clusters: A list of clusters to use for validation
        """
        seed = 5  # We shuffle with the same random seed so that training stays in training
        # and validation stays in validation
        Random(seed).shuffle(self.clusters)
        num = len(self.clusters)
        val_end = int(num*self.val_split)
        # Partition all clusters into a training and validation set
        train_clusters = self.clusters[val_end:]
        val_clusters = self.clusters[:val_end]
        if self.mode == 'test':
            self.logger.info(" {} clusters will be used for testing".format(
                len(train_clusters)))
        else:
            self.logger.info(" {} clusters will be used for training".format(
                len(train_clusters)))
            self.logger.info(" {} clusters will be used for validation".format(
                len(val_clusters)))
        self.train_clusters = train_clusters
        self.val_clusters = val_clusters
        return train_clusters, val_clusters

    def set_iter_clusters(self):
        """ Sets the clusters that will be returned by the iterator (train or val) """
        if self.mode == 'test':
            self.val_split = 0.0  # If we are testing, then we don't need a validation set
        train_clusters, val_clusters = self.get_clusters()
        if self.mode == 'training':
            self.iter_clusters = train_clusters
        elif self.mode == 'validation':
            self.iter_clusters = val_clusters
        elif self.mode == 'test':
            # If we are testing, the val_split is zero, and we test on the all clusters (ignore the naming as train)
            self.iter_clusters = train_clusters
        else:
            self.logger.error(
                "Mode {} is unsupported. Please use 'training' or 'validation' or 'test'".format(self.mode))
            exit(1)

    def __iter__(self):
        """ Iterator for returning batches of data in the form that MVSNet expects when training or validation
        Yields:
            images: Numpy array of stacked images. These are rescaled, cropped and centered
            cams: Numpy array of camera data for the above stack of images
            depth: Ground truth depth map. This is masked, rescaled and reshaped
        """

        if self.mode == 'training' or self.mode == 'validation':
            while True:
                for c in self.iter_clusters:
                    start = time.time()
                    images = c.images()
                    cams = c.cameras()
                    depth = c.masked_reference_depth()

                    # Crop, scale and center images
                    images, cams, depth = ut.scale_mvs_input(
                        images, cams, depth, c.rescale)
                    images, cams, depth = ut.crop_mvs_input(
                        images, cams, self.image_width, self.image_height, self.base_image_size, depth)
                    images = ut.center_images(images)
                    images = np.stack(images, axis=0)
                    cams = np.stack(cams, axis=0)

                    depth = ut.scale_and_reshape_depth(
                        depth, self.output_scale)

                    self.logger.debug(
                        'Load time: {}'.format(time.time() - start))
                    self.logger.debug('images shape: {}'.format(images.shape))
                    self.logger.debug('cams shape: {}'.format(cams.shape))
                    self.logger.debug('depth shape: {}'.format(depth.shape))
                    yield (images, cams, depth)

                    if self.flip_cams:
                        cams = ut.flip_cams(cams, self.depth_num)
                        yield (images, cams, depth)

        """ Iterator for returning batches of data in the form that MVSNet expects when testing
        Yields:
            output_images: Numpy array of stacked images that are rescaled to match output size of network
            input_images: Python list of images, rescaled, cropped and centered to match input size of network. The first
                            image in the list is the reference image
            output_cams: Numpy array of camera data that has been reformatted to match the output size of network
            image_index: The index of the reference image used for this cluster
        """
        if self.mode == 'test':
            while True:
                for c in self.iter_clusters:
                    start = time.time()
                    images = c.images()
                    cams = c.cameras()
                    # Crop, scale and center images
                    images, cams = ut.scale_mvs_input(
                        images, cams, scale=c.rescale)
                    cropped_images, cropped_cams = ut.crop_mvs_input(
                        images, cams, self.image_width, self.image_height, self.base_image_size)

                    # Scaled for input size
                    input_images = ut.copy_and_center_images(cropped_images)

                    # Scaled to the output size of network
                    output_images, output_cams = ut.scale_mvs_input(
                        cropped_images, cropped_cams, scale=self.output_scale)

                    output_images = np.stack(output_images, axis=0)
                    output_cams = np.stack(output_cams, axis=0)

                    image_index = c.ref_index
                    self.logger.debug(
                        'Load time: {}'.format(time.time() - start))
                    self.logger.debug(
                        'input image shape: {}'.format(input_images[0].shape))
                    self.logger.debug(
                        'output images shape: {}'.format(output_images.shape))
                    self.logger.debug(
                        'output cams shape: {}'.format(output_cams.shape))
                    self.logger.debug('image index: {}'.format(image_index))

                    c_dir = os.path.join(
                        self.sessions_dir, 'centered_images')
                    ut.mkdir_p(c_dir)
                    image_path = os.path.join(
                        c_dir, '{}.jpg'.format(image_index))
                    imageio.imsave(
                        image_path, input_images[0].astype(np.uint8))
                    yield (output_images, input_images, output_cams, image_index)

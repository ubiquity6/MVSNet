import os
from mvsnet.mvs_data_generation.mvs_cluster import Cluster
import mvsnet.mvs_data_generation.utils as ut
from mvsnet.utils import setup_logger
import random
import numpy as np
import imageio
import time
import pickle
import json
import logging
import tensorflow as tf
from tensorflow.python.lib.io import file_io
logging.basicConfig()

"""
Copyright 2019, Chris Heinrich, Ubiquity6.

The ClusterGenerator object serves up batches of data to be used for training a multi view stereo network. 

The data returned consists of multiple input images and their associated camera information, along with GT depth maps in 
the case of training, validation or benchmarking.

"""


class ClusterGenerator:
    def __init__(self, data_dir, view_num=3, image_width=1024, image_height=768, depth_num=256,
                 interval_scale=1, base_image_size=1, include_empty=False, mode='train', rescaling=True,
                 output_scale=0.25, flip_cams=True, sessions_frac=1.0, max_clusters_per_session=None, clear_cache=False):
        self.logger = setup_logger('ClusterGenerator')
        self.data_dir = data_dir
        self.mode = mode
        self.set_sessions_dir()
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
        self.rescaling = rescaling
        # Factor by which output is scaled relative to input
        self.output_scale = output_scale
        self.flip_cams = flip_cams
        # The sessions_fraction [0,1] is the fraction of all available sessions in sessions_dir
        self.sessions_frac = sessions_frac
        # max clusters per session is used if you don't want to train on all the clusters in a session
        self.max_clusters_per_session = max_clusters_per_session
        # Whether to clear the cache of pickled Cluster objects
        self.clear_cache = clear_cache
        self.parse_sessions()

    def set_sessions_dir(self):
        """ Sets which directory to use for creating clusters, depending on the mode of the data generator. If we are running inference then we 
        we expect the data_dir to be a single session, rather than a dir containing multiple sessions """
        if self.mode == 'train':
            self.sessions_dir = os.path.join(self.data_dir, 'train')
        elif self.mode == 'val':
            self.sessions_dir = os.path.join(self.data_dir, 'val')
        elif self.mode == 'test':
            self.sessions_dir = os.path.join(self.data_dir, 'test')
        elif self.mode == 'inference':
            self.sessions_dir = self.data_dir
        self.logger.debug(
            'Initializing cluster generator with sessions_dir {}'.format(self.sessions_dir))

    def parse_sessions(self):
        """ 
        Parses a directory of mvs training sessions and returns a 
        list of dictionaries describing visibility clusters from all sessions. If running
        in 'train' or 'val' mode then self.sessions_dir is expected to include multiple subdirectories
        with individual sessions. If running in 'inference' mode then self.sessions_dir is expected to include a single
        session, which is what will be used for computing depth  maps.

        Returns:
            clusters: A list of Cluster objects. See mvs_cluster.py for their declaration.
        """
        cache_path = os.path.join(self.sessions_dir, 'clusters.pickle')
        cache_exists = file_io.file_exists(cache_path)
        clusters = []
        if cache_exists and self.clear_cache is False and self.mode != 'inference':
            # load pickled clusters from cache
            self.logger.info(
                'Loading pickled cluster objects from {}'.format(cache_path))
            json_clusters = pickle.load(file_io.FileIO(cache_path, 'rb'))
            for data in json_clusters:
                clusters.append(Cluster(data['session_dir'], data['ref_index'], data['views'], data['min_depth'], data['max_depth'], data['view_num'],
                                        data['image_width'], data['image_height'], data['depth_num'], data['interval_scale']))
        else:
            if self.mode == 'inference':
                # If we are running inference then we only load clusters from one directory
                self.load_clusters(self.sessions_dir, clusters)
            else:
                self.logger.info(
                    'Metadata cache does not exist or clear_cache=True. Rebuilding metadata')
                sessions = [f for f in tf.gfile.ListDirectory(
                    self.sessions_dir) if not f.startswith('.') if not f.endswith('.txt')]
                sessions = sorted(sessions)
                total_sessions = len(sessions)
                self.logger.info(
                    'There are {} total sessions'.format(total_sessions))
                num_sessions = int(total_sessions * self.sessions_frac)
                self.logger.info('{} sessions will be used to {} because you set the fraction of sessions to use to {}'.format(
                    num_sessions, self.mode, self.sessions_frac))
                for s, session in enumerate(sessions[:num_sessions]):
                    session_dir = os.path.join(self.sessions_dir, session)
                    self.logger.debug(
                        'Parsing session dir {}'.format(session_dir))
                    try:
                        self.load_clusters(session_dir, clusters)
                    except Exception as e:
                        self.logger.debug(
                            'Failed to load clusters for session dir {} with exception {}'.format(session_dir, e))
                    if s % 50 == 0:
                        self.logger.info(
                            'Parsed {} / {} sessions'.format(s, num_sessions))
                self.cache_clusters(clusters, cache_path)

        if self.mode == 'train' or self.mode == 'val':
            random.shuffle(clusters)
        self.logger.info('{} clusters will be used to {}'.format(
            len(clusters), self.mode))
        self.clusters = clusters
        return clusters

    def cache_clusters(self, clusters, path):
        self.logger.info(
            'Pickling cluster objects to {}'.format(path))
        json_clusters = []
        for c in clusters:
            json_clusters.append(c.to_json())
        pickle.dump(json_clusters, file_io.FileIO(path, mode='wb'), -1)

    def load_clusters(self, session_dir, clusters):
        """ Loads all visibility clusters in a directory """
        with file_io.FileIO(os.path.join(session_dir, 'covisibility.json'), mode='r') as f:
            data = json.load(f)
        clusters_added = 0
        max_clusters = len(data)
        if self.max_clusters_per_session is not None:
            max_clusters = self.max_clusters_per_session

        for d in data:
            if not self.include_empty and not data[d]['views']:
                # Skip if there are no covisible views and we don't include empty
                pass
            elif clusters_added < max_clusters:
                cluster = Cluster(session_dir, int(d), data[d]['views'], data[d]['min_depth'],
                                  data[d]['max_depth'], self.view_num, self.image_width, self.image_height, self.depth_num, self.interval_scale)
                clusters.append(cluster)
                clusters_added += 1

    def __iter__(self):
        """ Iterator for returning batches of data in the form that MVSNet expects when training or validation
        Yields:
            images: Numpy array of stacked images. These are rescaled, cropped and centered
            cams: Numpy array of camera data for the above stack of images
            depth: Ground truth depth map. This is masked, rescaled and reshaped
        """

        if self.mode == 'train' or self.mode == 'val':
            while True:
                for c in self.clusters:
                    # We wrap this in a try/except block because we don't want to end execution of the program
                    # just because a cluster or two may have bad data
                    try:
                        start = time.time()
                        images = c.images()
                        cams = c.cameras()
                        depth = c.masked_reference_depth()
                        load_time = time.time() - start
                        self.logger.debug(
                            'Cluster data load time: {}'.format(load_time))

                        # Crop, scale and center images
                        images, cams, depth = ut.scale_mvs_input(
                            images, cams, depth, c.rescale)
                        images, cams, depth = ut.crop_mvs_input(
                            images, cams, self.image_width, self.image_height, self.base_image_size, depth)
                        images = ut.center_images(images)
                        images = np.stack(images, axis=0)

                        # output_depth and output_cams are copies of depth and cams that are downsampled
                        # by output_scale, as these downsampled copies are used for computation at certain stages
                        rescaled_depth = ut.scale_and_reshape_depth(
                            depth, self.output_scale)
                        depth = ut.reshape_depth(depth)
                        cams = ut.scale_mvs_camera(
                            cams, scale=self.output_scale)
                        cams = np.stack(cams, axis=0)

                        self.logger.debug(
                            'Cluster transformation time: {}'.format(time.time() - start - load_time))

                        self.logger.debug(
                            'Total cluster preparation time: {}'.format(time.time() - start))
                        self.logger.debug(
                            'images shape: {}'.format(images.shape))
                        self.logger.debug('cams shape: {}'.format(cams.shape))
                        self.logger.debug(
                            'Full depth shape: {}'.format(depth.shape))
                        self.logger.debug(
                            'Rescaled depth shape: {}'.format(rescaled_depth.shape))
                        self.logger.debug(
                            'Reference index: {}'.format(c.ref_index))
                        self.logger.debug('Cluster indices: {}. Session dir: {}'.format(
                            c.indices, c.session_dir))
                        self.logger.debug('Cluster generator mode: {} '.format(
                            self.mode))
                        yield (images, cams, rescaled_depth, depth)

                        if self.flip_cams:
                            cams = ut.flip_cams(cams, self.depth_num)
                            yield (images, cams, rescaled_depth, depth)

                    except Exception as e:
                        self.logger.warn('Cluster with indices: {} at dir: {} failed to load with error: "{}". Skipping!'.format(
                            c.indices, c.session_dir, e))
                        continue

        """ Iterator for returning batches of data in the form that MVSNet expects when testing
        Yields:
            output_images: Numpy array of stacked images that are rescaled to match output size of network
            input_images: Python list of images, rescaled, cropped and centered to match input size of network. The first
                            image in the list is the reference image
            output_cams: Numpy array of camera data that has been reformatted to match the output size of network
            image_index: The index of the reference image used for this cluster
        """
        if self.mode == 'inference' or self.mode == 'test':
            while True:
                for c in self.clusters:
                    start = time.time()
                    images = c.images()
                    cams = c.cameras()
                    # Crop, scale and center images
                    if self.mode == 'test':
                        # We also need to retrieve GT depth data if we are benchmarking
                        depth = c.masked_reference_depth()
                        images, cams, depth = ut.scale_mvs_input(
                            images, cams, depth, c.rescale)
                        cropped_images, cropped_cams, depth = ut.crop_mvs_input(
                            images, cams, self.image_width, self.image_height, self.base_image_size, depth)
                        depth = ut.reshape_depth(depth)

                    else:
                        images, cams = ut.scale_mvs_input(
                            images, cams, scale=c.rescale)
                        cropped_images, cropped_cams = ut.crop_mvs_input(
                            images, cams, self.image_width, self.image_height, self.base_image_size)
                    # Full cams are scaled to input image resolution
                    full_cams = np.stack(cropped_cams, axis=0)
                    # Scaled for input size
                    input_images = ut.copy_and_center_images(cropped_images)
                    # Scaled to the output size of network
                    # Scaled cams are used for the differential homography warping
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

                    if self.mode == 'test':
                        yield (output_images, input_images, output_cams, full_cams, depth, image_index, c.session_dir)
                    else:
                        yield (output_images, input_images, output_cams, full_cams, image_index)

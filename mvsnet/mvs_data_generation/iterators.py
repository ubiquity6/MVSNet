from cluster_generator import ClusterGenerator
from tensorflow.python.lib.io import file_io
import tensorflow as tf
import os
import argparse
import time
import logging
import utils as ut
logging.basicConfig()

"""
Copyright 2019, Chris Heinrich, Ubiquity6.
"""


class DataIterator:
    def __init__(self, sessions_dir, view_num, image_width=1024, image_height=768, depth_num=256,
                 interval_scale=1, base_image_size=1, include_empty=False, mode='training', val_split=0.1, rescaling=True, output_scale=0.25, flip_cams=True, num_generators=1, batch_size=1):
        # Setup logger

        self.logger = logging.getLogger('ParallelIterator')
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
        self.num_generators = num_generators
        self.batch_size = batch_size

    def generator(self):
        train_gen = ClusterGenerator(self.sessions_dir, self.view_num, self.image_width, self.image_height,
                                     self.depth_num, self.interval_scale, self.base_image_size, mode=self.mode, flip_cams=self.flip_cams, output_scale=self.output_scale, rescaling=self.rescaling, val_split=self.val_split)
        return iter(train_gen)

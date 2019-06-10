import logging
import tensorflow as tf 
import os

"""
A small library of helper functions sued through the mvsnet package 
"""

def set_log_level(logger):
    """ Grabs log level from command line """
    try:
        level = os.environ['LOG_LEVEL'].upper()
        exec('logger.setLevel(logging.{})'.format(level))
    except Exception as e:
        logger.setLevel(logging.INFO)

def setup_logger(name):
    """ Sets up a logger, grabbing log_level from command line """
    logging.basicConfig()
    logger = logging.getLogger(name)
    try:
        set_log_level(logger)
    except Exception as e:
        logger.setLevel(logging.INFO)
        logger.warn('Failed to set log level with exception {}'.format(e))
    return logger

    
def init_session():
    """ Returns tf global vars initializer and sets the config """
    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.inter_op_parallelism_threads = 0
    config.intra_op_parallelism_threads = 0
    return init_op, config

def mkdir_p(dir_path):
    """ Makes the directory dir_path if it doesn't exist """
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
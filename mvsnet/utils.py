import logging 
import os


def set_log_level(logger):
    if 'LOG_LEVEL' in os.environ:
        level = os.environ['LOG_LEVEL'].upper()
        exec('logger.setLevel(logging.{})'.format(level))
    else:
        logger.setLevel(logging.INFO)

def setup_logger(name):
    logging.basicConfig()
    logger = logging.getLogger(name)
    set_log_level(logger)
    return logger
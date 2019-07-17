from distutils.core import setup
from setuptools import find_packages
import os

# Package Metadata
NAME = 'MVSNet'
VERSION = '0.1.1'


def ml_engine():
    """ Checks whether this package is being installed on a Google AI Platform machine """
    if 'CLOUD_ML_JOB_ID' in os.environ:
        return True
    else:
        return False


def required_packages_ml_engine():
    # ml-engine already has tensorflow-gpu installed, and a few other
    # packages require different versions on ml-engine
    PACKAGES = [
        'progressbar2>=3.0',
        'numpy>=1.13',
        'opencv-python>=3.2',
        'scikit-learn>=0.18',
        'scipy>=0.18',
        'matplotlib>=1.5',
        'Pillow>=3.1.2',
        'imageio',
        'wandb',
    ]
    return PACKAGES


def required_packages():
    if ml_engine():
        return required_packages_ml_engine()
    else:
        PACKAGES = [
            'progressbar2==3.0.1',
            'numpy==1.16.2',
            'opencv-python-headless==4.1.0.25',
            'scikit-learn==0.18',
            'scipy==0.18',
            'matplotlib==1.5',
            'tensorflow==1.12.0',
            'funcsigs==1.0.2',
            'Pillow==6.1.0',
            'imageio==2.5.0',
            'wandb==0.8.4',
        ]
        return PACKAGES


setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(exclude=['datasets*', 'scripts*']),
    install_requires=required_packages(),
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
)

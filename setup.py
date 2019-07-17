from distutils.core import setup
from setuptools import find_packages

# Package Metadata
NAME='MVSNet'
VERSION='0.1.2'

def required_packages():
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
        'Click==0.7',
    ]
    return PACKAGES

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(exclude=['datasets*','scripts*']),
    install_requires=required_packages(),
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
)

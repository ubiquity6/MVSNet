from distutils.core import setup
from setuptools import find_packages

# Package Metadata
NAME='MVSNet'
VERSION='0.1.0'

def required_packages():
    PACKAGES = [
        'progressbar2>=3.0',
        'numpy>=1.13',
        'opencv-python-headless==4.1.0.25',
        'scikit-learn>=0.18',
        'scipy>=0.18',
        'matplotlib>=1.5',
        'Pillow>=3.1.2',
        'imageio',
        'wandb',
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

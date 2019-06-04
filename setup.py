from distutils.core import setup
from setuptools import find_packages

# Package Metadata

NAME='MVSNet'
VERSION='0.4.0'




def required_packages():
    # place dependencies here
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
        'tensorflow==1.13.1',
        'tensorflow-estimator==1.10.12',
    ]
    return PACKAGES

setup(
    name=NAME,
    version=VERSION,
    packages=['mvsnet', 'mvsnet/cnn_wrapper','mvsnet/mvs_data_generation'],
    #packages=find_packages(),
    install_requires=required_packages(),
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
)

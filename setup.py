from distutils.core import setup
from setuptools import find_packages


def required_packages():
    # place dependencies that don't care about GPU vs CPU here
    PACKAGES = [
        'progressbar2>=3.0',
        'numpy>=1.13',
        'opencv-python>=3.2',
        'scikit-learn>=0.18',
        'scipy>=0.18',
        'matplotlib>=1.5',
        'Pillow>=3.1.2',
        'imageio'
    ]
    return PACKAGES

setup(
    name='MVSNet',
    version='0.3dev',
    packages=['mvsnet', 'mvsnet/cnn_wrapper','mvsnet/mvs_data_generation'],
    install_requires=required_packages(),
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
)

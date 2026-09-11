from distutils.core import setup
from setuptools import find_packages
from setuptools.command.install import install
import os

setup(
    name='graspnetAPI',
    version='1.2.11',
    description='graspnet API',
    author='Hao-Shu Fang, Chenxi Wang, Minghao Gou',
    author_email='gouminghao@gmail.com',
    url='https://graspnet.net',
    packages=find_packages(),
    install_requires=[
    ]
)

#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path
from glob import glob

this_directory = path.abspath(path.dirname(__file__))

setup(
    name='clwe-transfer',
    description='Implementation of https://arxiv.org/abs/2103.06689',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        "dvc",
        "numpy",
        "fasttext",
        "sacrebleu",
        "sacremoses",
        "scipy"
    ],
    scripts=glob(this_directory + '/bin/*'),
)

#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="spint",
    version="1.0.0",
    description="SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding",
    author="Trung Le, Hao Fang, Jingyuan Li, Tung Nguyen, Lu Mi, Amy Orsborn, Uygar Sumbul, Eli Shlizerman",
    author_email="trungle@uw.edu",
    url="https://github.com/shlizee/SPINT",
    license="BSD-3-Clause",
    install_requires=[
        "lightning",
        "hydra-core",
        "falcon-challenge",
        "scipy",
        "pynwb",
        "scikit-learn",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)

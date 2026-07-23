from setuptools import find_packages, setup

setup(
    name="streaming-calibration-exp",
    version="0.1.0",
    description="SPINT streaming calibration encoder experiments",
    packages=find_packages(),
    install_requires=[
        "lightning",
        "hydra-core",
        "falcon-challenge",
        "scipy",
        "rootutils",
        "torchmetrics",
    ],
)

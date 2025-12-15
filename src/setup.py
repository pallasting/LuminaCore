from setuptools import setup, find_packages

setup(
    name="luminaflow",
    version="0.1.0-alpha",
    description="PyTorch SDK for LuminaCore Photonic Computing Architecture",
    author="LuminaCore Team",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy",
        "matplotlib"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
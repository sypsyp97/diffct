from setuptools import setup, find_packages

setup(
    name="diffct",
    version="1.0.0",
    description="A CUDA-based library for computed tomography (CT) projection and reconstruction with differentiable operators",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Yipeng Sun",
    author_email="yipeng.sun@fau.de",  # Replace with actual email if available
    url="https://github.com/sypsyp97/differentiable-ct-reconstruction",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "numba",
        "torch",
    ],
    license="Apache 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
    ],
    python_requires=">=3.10",
)
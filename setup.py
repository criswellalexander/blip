#!/usr/bin/env python

from setuptools import setup

with open("README.md", "r") as rme:
    long_description = rme.read()

setup(
    name="blip",
    description="A Bayesian inference package for characterizing stochastic backgrounds and foregrounds with LISA.",
    long_description=long_description,
    url="https://github.com/sharanbngr/blip",
    author="Sharan Banagiri, Alexander Criswell, and others",
    author_email="sharan.banagiri@gmail.com",
    license="MIT",
    packages=["blip",
        "blip.src",
        "blip.tools",
        ],
    package_dir={"blip":"blip"},
    scripts=["blip/run_blip"],
    install_requires=[
        "numpy>=2.0",
        "matplotlib",
        "healpy",
        "scipy",
        "astropy",
        "pandas",
        "chainconsumer==0.34.0",
        "sympy",
        "legwork",
        "dill",
        "dynesty",
        "emcee",
        "numpyro",
        "jax"
    ],
    extras_require={
            "gpu":["jax[cuda12]"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)



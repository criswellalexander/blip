#  BLIP: Bayesian LISA Inference Package

This is a fully Bayesian Python package for detecting/characterizing stochastic gravitational wave backgrounds and foregrounds with LISA. It is designed to support flexible, modular analyses of multiple isotropic and anisotropic stochastic signals, with a variety of source morphologies.


1) We recommend creating a dedicated conda environment for BLIP. Conda is a common python virtual environment manager; if you already have Conda, start at step 2; otherwise [install conda.](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

2) Create an environment. We require Python >= 3.10, and recommend 3.12:

`conda create --name blip-env python=3.12`


3) Activate it via

`conda activate blip-env`

4) You can now install the package via pip by running

`pip install -e .`

in this directory. The `-e` option performs an editable install, so that editing the code (either yourself or via pullng updates from GitHub) will update your installation, without requiring you to run the installation process over again. If this is not desired, the `-e` can be omitted.

4a) If you wish to run BLIP with GPU-acceleration, instead run

`pip install -e .[gpu]`

which will install JAX with CUDA 12 dependencies.

## TEMPORARY FIX FOR CHAINCONSUMER
We require chainconsumer 0.34.0, as the author of that code has revamped it in more recent versions to be significantly clunkier. We will soon be depreciating chainconsumer in favor of corner or our own implementation. For the moment, however, chainconsumer 0.34.0 has a breaking import. To resolve this, follwing the installation process, navigate to (e.g.) [your conda path]/envs/blip-env/lib/python3.12/site-packages/chainconsumer/analysis.py. On line 4, replace
`from scipy.integrate import simps`
with
`from scipy.integrate import simpson as simps`

You should now be ready to go! To run BLIP, you only need to provide a configuration file. In this directory, you will find params_default.ini, a pre-constructed config file with reasonable settings and accompanying parameter explanations.

To run, call

`run_blip params_default.ini`

This will (by default) inject and recover a power law isotropic SGWB, with LISA detector noise at the level specified in the LISA proposal (Amaro-Seoane et al., 2017), for 3 months of data.

Two other helpful parameter files are also included: params_test.ini, which has settings ideal for (more) rapid code testing, and params_simple.ini, which only includes the bare bones, minimal necessary settings for BLIP to run.

Posterior plots will be automatically created in the specified output directory, along with some diagnostics. All statistical model information is saved in Model.pickle; all information used to perform the injection is likewise saved in Injection.pickle. The posterior samples are saved to post_samples.txt.

More details can be found in [the code documentation](https://blip.readthedocs.io/en/latest/).
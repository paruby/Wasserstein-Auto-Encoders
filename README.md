# Wasserstein-Auto-Encoders

This repository contains code relating to our arxiv paper "On the Latent Space
of Wasserstein Auto-Encoders" https://arxiv.org/abs/1802.03761

The code is still being cleaned up, so keep in mind that this repository may
change significantly.



# Requirements
Python 3.5+

# Structure of repository

wae.py contains code to build wae model. Model will actually be built by calling initialisation functions from other files, eg models.py.
models.py contains specific encoder, decoder and priors
utils.py contains plotting tools and other tools involving loading and saving data
config.py contains example configurations of options

# To train a model

Call run.py with a variety of possible options seen in argparse at the top of run.py
Calling just a dataset will give default settings for that experiment e.g.

python run.py --datasetset dsprites

These can be overwritten by using other flags, e.g.

python run.py --datasetset dsprites --experiment_path dsprites/exp10 --z_dim 12

# To load a model

To open an interactive session with a saved model, call load.py specifying the experiment_path for a saved model, e.g.

python -i load.py dsprites/exp10


Todos:
- mean/min/max log variance logging
- plotting functions on model object for ease of visualising loaded models
- adding dcgan architecture to harmonise with ilya's code

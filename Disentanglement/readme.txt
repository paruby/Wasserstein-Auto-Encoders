Each experiment that is run will produce a folder of results in the folder
dsprites. To run, invoke e.g.

python execute_all.py dsprites 16 400 3.0 100

Explanation of parameters:

dsprites - dataset name. In the current version of code published on github, you have no choice but to use dsprites
16 - latent space dimensionality
400 - Lambda for MMD divergence term in loss function
3.0 - Lambda_L1 for L1 penalty on log-variances
100 - Batch size. Note that MMD memory requirements scale quadratically with batch size. 


Explanation of output:

average_logvars.log - average log-variances along each dimension across the
current train and test minibatches. Useful for seeing if dimensions are being
'switched off' by being filled with noise, corresponding to logvar ~ 0
disentanglement4.txt  - calculates disentanglement metric of beta-VAE paper on
4 variable disentanglement task
disentanglement5.txt  - same as above but with 5 variable disentanglement task
output/plots  - plots of train and test reconstructions, random samples,
interpolations and latent axis traversals
parameters.txt  - record of the parameters used in experiment
README.txt  - recorded message to describe why the experiment was done (blank)
saved_model - saved copy of model by tf.saver  
test_losses.log  - each component of the loss function for the test minibatch 
training_losses.log - each component of the loss for train minibatch


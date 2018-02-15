
assert opts["encoder_architecture"] in ['small_convolutional_celebA', 'FC_dsprites']
assert opts['z_mean_activation'] in ['tanh' or None]
assert opts['encoder_distribution'] in ['deterministic', 'gaussian', 'uniform']
assert opts['logvar-clipping'] in [None] or (len(opts['logvar-clipping']) == 2 and all([type(i) is int for i in opts['logvar-clipping']])
assert opts['z_prior'] in ['gaussian', 'uniform']
assert opts['loss_reconstruction'] in ['bernoulli', 'L2_squared']
assert opts['regulariser'] in ['VAE', 'beta_VAE', 'WAE_MMD'] # either KL divergence of VAE or divergence of WAE
if opts['regulariser'] == 'beta_VAE':
    assert type(opts['beta']) is float
if opts['regulariser'] == 'WAE_MMD':
    assert type(opts['lambda_imq']) is float
    assert type(opts['IMQ_length_params']) is list # parameters should be scaled according to z_dim
    assert all(type(i) is float for i in opts['IMQ_length_params'])
assert opts['z_logvar_regularisation'] in [None, "L1", "L2_squared"]
assert opts['optimizer'] in ['adam']
if opts['optimizer'] == 'adam':
    assert type(opts['learning_rate']) is float

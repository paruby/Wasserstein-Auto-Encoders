

assert type(opts['z_dim']) is int
assert opts['print_log_information'] in [True, False]
assert (type(opts['make_pictures_every']) is int) or (opts['make_pictures_every'] is None)
assert type(opts['plot_axis_walks']) is bool
if opts['plot_axis_walks'] is True: #if z_dim >> 10, plotting axis walks will be long
    opts['axis_walk_range'] > 0
assert type(opts['plot_losses']) is bool
if opts['plot_losses'] is True:
    assert opts['print_log_information'] is True
assert type(opts['batch_size']) is int
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
    assert type(opts['learning_rate_schedule']) is list
    assert all([type(l) is tuple and len(l)==2 for l in opts['learning_rate_schedule']])
    assert all([opts['learning_rate_schedule'][i] < opts['learning_rate_schedule'][i+1] for i in range(len(opts['learning_rate_schedule'])-1)])
    # opts['learning_rate_schedule'] = [(learning_rate, iteration # that this is valid for)]
    # e.g. opts['learning_rate_schedule'] = [(1e-4, 20000), (3e-5, 40000), (1e-5, 60000)]

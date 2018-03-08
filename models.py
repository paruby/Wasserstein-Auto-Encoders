import tensorflow as tf
import numpy as np


def encoder_init(model):
    with tf.variable_scope('encoder'):
        if model.opts['encoder_architecture'] == 'small_convolutional_celebA':
            _encoder_small_convolutional_celebA_init(model)
        elif model.opts['encoder_architecture'] == 'FC_dsprites':
            _encoder_FC_dsprites_init(model)
        elif model.opts['encoder_architecture'] == 'dcgan':
            _dcgan_encoder(model)
        _z_sample_init(model)


def prior_init(model):
    if model.opts['z_prior'] == 'gaussian':
        model.z_prior_sample = tf.random_normal(shape=tf.shape(model.z_mean),
                                                name="z_prior_sample")
    elif model.opts['z_prior'] == 'uniform':
        noise = tf.random_uniform(shape=tf.shape(model.z_mean))
        model.z_prior_sample = tf.multiply((noise-0.5), 2,
                                            name="z_prior_sample")

def decoder_init(model):
    with tf.variable_scope('decoder'):
        if model.opts['decoder_architecture'] == 'small_convolutional_celebA':
            _decoder_small_convolutional_celebA_init(model)
        elif model.opts['decoder_architecture'] == 'FC_dsprites':
            _decoder_FC_dsprites_init(model)
        elif model.opts['decoder_architecture'] in ['dcgan', 'dcgan_mod']:
            _dcgan_decoder(model)

def loss_init(model):
    all_losses = []
    if model.opts['loss_reconstruction'] == 'bernoulli':
        model.loss_reconstruction = tf.reduce_mean(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=model.x_logits,
                                                    labels=model.x_flattened),
                                                    axis=1),
                                                    name="loss_reconstruction")
    elif model.opts['loss_reconstruction'] == 'L2_squared':
        model.loss_reconstruction = tf.reduce_mean(tf.reduce_sum(
            tf.square(tf.nn.sigmoid(model.x_logits) - model.x_flattened), axis=1),
            name="loss_reconstruction")

    elif model.opts['loss_reconstruction'] == 'L2_squared+adversarial':
        if 'adv_cost_lambda' not in model.opts:
            adv_cost_lambda = 1.0
        with tf.variable_scope('adversarial_cost'):
            out_im = tf.nn.sigmoid(model.x_logits_img_shape)
            n_filters = model.opts['adversarial_cost_n_filters']
            kernel_size = model.opts['adversarial_cost_kernel_size']
            fake_img_repr  = tf.layers.conv2d(out_im,
                                              filters=n_filters,
                                              strides=1,
                                              kernel_size=[kernel_size,kernel_size],
                                              name='adv_cost_repr')
            real_img_repr  = tf.layers.conv2d(model.input,
                                              filters=n_filters,
                                              strides=1,
                                              kernel_size=[kernel_size,kernel_size],
                                              name='adv_cost_repr',
                                              reuse=True)
            sq_diff = (tf.nn.sigmoid(real_img_repr) - tf.nn.sigmoid(fake_img_repr))**2
            model.adv_cost_loss = tf.reduce_sum(tf.reduce_mean(sq_diff, axis=0), name='adv_cost_loss')
            l2_sq_loss = tf.reduce_sum(tf.reduce_mean((out_im - model.input)**2, axis=0))
            model.loss_reconstruction = tf.add(model.adv_cost_loss, l2_sq_loss, name='loss_reconstruction')

    elif model.opts['loss_reconstruction'] == 'L2_squared+adversarial+l2_filter':
        if 'adv_cost_lambda' not in model.opts:
            adv_cost_lambda = 1.0
        else:
            adv_cost_lambda = model.opts['adv_cost_lambda']
        with tf.variable_scope('adversarial_cost'):
            out_im = tf.nn.sigmoid(model.x_logits_img_shape)
            n_filters = model.opts['adversarial_cost_n_filters']
            kernel_size = model.opts['adversarial_cost_kernel_size']
            fake_img_repr  = tf.layers.conv2d(out_im,
                                              filters=n_filters,
                                              strides=1,
                                              kernel_size=[kernel_size,kernel_size],
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                              name='adv_cost_repr')
            real_img_repr  = tf.layers.conv2d(model.input,
                                              filters=n_filters,
                                              strides=1,
                                              kernel_size=[kernel_size,kernel_size],
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                              name='adv_cost_repr',
                                              reuse=True)
            sq_diff = (real_img_repr - fake_img_repr)**2
            sq_diff = tf.reduce_mean(sq_diff, axis=[0,3]) # mean over batch and channels
            model.adv_cost_loss = tf.reduce_sum(sq_diff, name='adv_cost_loss')
            l2_sq_loss = tf.reduce_sum(tf.reduce_mean((out_im - model.input)**2, axis=0))
            model.loss_reconstruction = tf.add(adv_cost_lambda * model.adv_cost_loss, l2_sq_loss, name='loss_reconstruction')

    elif model.opts['loss_reconstruction'] == 'L2_squared+adversarial+l2_norm':
        if 'adv_cost_lambda' not in model.opts:
            adv_cost_lambda = 1.0
        else:
            adv_cost_lambda = model.opts['adv_cost_lambda']
        with tf.variable_scope('adversarial_cost'):
            out_im = tf.nn.sigmoid(model.x_logits_img_shape)
            channels = int(out_im.get_shape()[-1])
            n_filters = model.opts['adversarial_cost_n_filters']
            kernel_size = model.opts['adversarial_cost_kernel_size']
            if 'adv_cost_normalise_filter' in model.opts:
                if model.opts['adv_cost_normalise_filter'] is True:
                    w = tf.get_variable('adv_filter',
                                        [kernel_size**2, channels, n_filters],
                                        initializer=tf.truncated_normal_initializer(stddev=0.01))
                    w = tf.nn.l2_normalize(w, 0)
                    w = tf.reshape(w, [kernel_size, kernel_size, channels, n_filters])
            else:
                w = tf.get_variable('adv_filter',
                                    [kernel_size, kernel_size, channels, n_filters],
                                    initializer=tf.truncated_normal_initializer(stddev=0.01))
                w = tf.nn.l2_normalize(w, 2)

            bias = tf.get_variable('adv_bias',
                                   [n_filters],
                                   initializer=tf.constant_initializer(0.001))

            fake_img_repr = tf.nn.conv2d(out_im, w, strides=[1,1,1,1], padding="SAME")
            fake_img_repr = tf.nn.bias_add(fake_img_repr, bias)

            real_img_repr = tf.nn.conv2d(model.input, w, strides=[1,1,1,1], padding="SAME")
            real_img_repr = tf.nn.bias_add(real_img_repr, bias)

            sq_diff = (real_img_repr - fake_img_repr)**2
            sq_diff = tf.reduce_mean(sq_diff, axis=[0,3]) # mean over batch and channels
            model.adv_cost_loss = tf.reduce_sum(sq_diff, name='adv_cost_loss')
            l2_sq_loss = tf.reduce_sum(tf.reduce_mean((out_im - model.input)**2, axis=0))
            model.loss_reconstruction = tf.add(adv_cost_lambda * model.adv_cost_loss, l2_sq_loss, name='loss_reconstruction')

    elif model.opts['loss_reconstruction'] == 'normalised_conv_adv':
        if 'adv_cost_lambda' not in model.opts:
            # weighting of l2 difference between featurised versions of images
            adv_cost_lambda = 1.0
        else:
            adv_cost_lambda = model.opts['adv_cost_lambda']
        if 'l2_lambda' not in model.opts:
            # weighting of pixelwise l2 loss
            l2_lambda = 1.0
        else:
            l2_lambda = model.opts['l2_lambda']
        if 'patch_classifier_lambda' not in model.opts:
            # weighting of patch classifier
            patch_classifier_lambda = 1.0
        else:
            patch_classifier_lambda = model.opts['patch_classifier_lambda']


        with tf.variable_scope('adversarial_cost'):
            out_im = tf.nn.sigmoid(model.x_logits_img_shape)
            real_im = model.input
            if 'adv_use_sq_features' in model.opts:
                if model.opts['adv_use_sq_features'] is True:
                    out_im_sq = out_im**2
                    out_input = tf.concat([out_im, out_im_sq], axis=-1)
                    real_im_sq = real_im**2
                    real_input = tf.concat([real_im, real_im_sq], axis=-1)
            else:
                real_input = real_im
                out_input = out_im

            height = int(out_input.get_shape()[1])
            width = int(out_input.get_shape()[2])
            channels = int(out_input.get_shape()[-1])
            n_filters = model.opts['adversarial_cost_n_filters']
            adversarial_cost = 0

            for kernel_size in [3,4,5]:
                w = tf.get_variable('adv_filter_%d' % kernel_size,
                                    [(kernel_size**2) * channels, n_filters],
                                    initializer=tf.truncated_normal_initializer(stddev=0.01))
                w = tf.nn.l2_normalize(w, 0)
                w = tf.reshape(w, [kernel_size, kernel_size, channels, n_filters])


                bias = tf.get_variable('adv_bias_%d' % kernel_size,
                                       [n_filters],
                                       initializer=tf.constant_initializer(0.001))

                fake_img_repr = tf.nn.conv2d(out_input, w, strides=[1,1,1,1], padding="SAME")
                fake_img_repr = tf.nn.bias_add(fake_img_repr, bias)

                real_img_repr = tf.nn.conv2d(real_input, w, strides=[1,1,1,1], padding="SAME")
                real_img_repr = tf.nn.bias_add(real_img_repr, bias)

                sq_diff = (real_img_repr - fake_img_repr)**2
                sq_diff = tf.reduce_mean(sq_diff, axis=[0,3]) # mean over batch and channels
                sq_diff = tf.reduce_sum(sq_diff)

                adversarial_cost += adv_cost_lambda * sq_diff

                real_img_repr = lrelu(0.1, real_img_repr)
                fake_img_repr = lrelu(0.1, fake_img_repr)
                w = tf.get_variable('adv_filter_layer2',[1, 1, n_filters, 1],
                                    initializer=tf.truncated_normal_initializer(stddev=0.01))

                real_img_repr = tf.nn.conv2d(real_img_repr, w, strides=[1,1,1,1], padding="SAME")
                real_img_logits = tf.reshape(real_img_repr, shape=[-1, height*width])
                fake_img_repr = tf.nn.conv2d(fake_img_repr, w, strides=[1,1,1,1], padding="SAME")
                fake_img_logits = tf.reshape(fake_img_repr, shape=[-1, height*width])

                patch_classification_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                                            logits=fake_img_logits, labels=tf.zeros_like(fake_img_logits))
                patch_classification_real = tf.nn.sigmoid_cross_entropy_with_logits(
                                            logits=real_img_logits, labels=tf.ones_like(real_img_logits))

                patch_classification_real = tf.reduce_mean(patch_classification_real)
                patch_classification_fake = tf.reduce_mean(patch_classification_fake)

                patch_classification_loss = patch_classification_fake + patch_classification_real

                adversarial_cost += patch_classifier_lambda * patch_classification_loss

            model.adv_cost_loss = tf.add(adversarial_cost, 0, name='adv_cost_loss')
            l2_sq_loss = l2_lambda * tf.reduce_sum(tf.reduce_mean((out_im - real_im)**2, axis=0))
            model.loss_reconstruction = tf.add(model.adv_cost_loss, l2_sq_loss, name='loss_reconstruction')


    elif model.opts['loss_reconstruction'] == 'L2_squared+multilayer_conv_adv':
        if 'adv_cost_lambda' not in model.opts:
            adv_cost_lambda = 1.0
        else:
            adv_cost_lambda = model.opts['adv_cost_lambda']
        if 'adv_cost_nlayers' not in model.opts:
            adv_cost_nlayers = 3
        else:
            adv_cost_nlayers = model.opts['adv_cost_nlayers']
        with tf.variable_scope('adversarial_cost'):
            out_im = tf.nn.sigmoid(model.x_logits_img_shape)
            n_filters = model.opts['adversarial_cost_n_filters']
            kernel_size = model.opts['adversarial_cost_kernel_size']
            layer_fake = out_im
            layer_real = model.input
            for i in range(adv_cost_nlayers-1):
                layer_fake  = lrelu(0.1, tf.layers.conv2d(layer_fake,
                                                  filters=n_filters,
                                                  strides=1,
                                                  kernel_size=[kernel_size,kernel_size],
                                                  name='adv_cost_repr_%d' % i))
                layer_real  = lrelu(0.1, tf.layers.conv2d(layer_real,
                                                  filters=n_filters,
                                                  strides=1,
                                                  kernel_size=[kernel_size,kernel_size],
                                                  name='adv_cost_repr_%d' % i,
                                                  reuse=True))

            fake_img_repr  = tf.layers.conv2d(layer_fake,
                                              filters=n_filters,
                                              strides=1,
                                              kernel_size=[kernel_size,kernel_size],
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                              name='adv_cost_repr')
            real_img_repr  = tf.layers.conv2d(layer_real,
                                              filters=n_filters,
                                              strides=1,
                                              kernel_size=[kernel_size,kernel_size],
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                              name='adv_cost_repr',
                                              reuse=True)
            sq_diff = (real_img_repr - fake_img_repr)**2
            sq_diff = tf.reduce_mean(sq_diff, axis=[0,3]) # mean over batch and channels
            model.adv_cost_loss = tf.reduce_sum(sq_diff, name='adv_cost_loss')
            model.adv_cost_loss = tf.reduce_mean(sq_diff, name='adv_cost_loss')
            l2_sq_loss = tf.reduce_sum(tf.reduce_mean((out_im - model.input)**2, axis=0))
            model.loss_reconstruction = tf.add(adv_cost_lambda * model.adv_cost_loss, l2_sq_loss, name='loss_reconstruction')

    elif model.opts['loss_reconstruction'] == 'patch_moments':
        # cost is the difference between mean and variance of local patches.
        # Take squared difference between mean and variance for patches and sum up
        out_im = tf.nn.sigmoid(model.x_logits_img_shape)
        out_im_sq = out_im**2
        real_im = model.input
        real_im_sq = model.input**2

        height, width, channels = [int(out_im.get_shape()[i]) for i in range(1,4)]

        if model.opts['adversarial_cost_kernel_size'] != -1:
            kernel_size = model.opts['adversarial_cost_kernel_size']
            w_sum = tf.eye(num_rows=channels, num_columns=channels, batch_shape=[kernel_size * kernel_size])
            w_sum = tf.reshape(w_sum, [kernel_size, kernel_size, channels, channels])
            w_sum = w_sum / (kernel_size*kernel_size)

            out_im_mean = tf.nn.conv2d(out_im, w_sum, strides=[1,1,1,1], padding='VALID')
            real_im_mean = tf.nn.conv2d(real_im, w_sum, strides=[1,1,1,1], padding='VALID')

            out_im_var = tf.nn.conv2d(out_im_sq, w_sum, strides=[1,1,1,1], padding='VALID') - out_im_mean**2
            real_im_var = tf.nn.conv2d(real_im_sq, w_sum, strides=[1,1,1,1], padding='VALID') - real_im_mean**2

            sq_var_diff = tf.reduce_mean((out_im_var - real_im_var)**2, axis=0)
            sq_var_diff = tf.reduce_sum(sq_var_diff)
        else:
            # sum patch variances using kernels of size 3 4 and 5
            var_diff = 0
            for kernel_size in [3, 4, 5]:
                w_sum = tf.eye(num_rows=channels, num_columns=channels, batch_shape=[kernel_size * kernel_size])
                w_sum = tf.reshape(w_sum, [kernel_size, kernel_size, channels, channels])
                w_sum = w_sum / (kernel_size*kernel_size)

                out_im_mean = tf.nn.conv2d(out_im, w_sum, strides=[1,1,1,1], padding='VALID')
                real_im_mean = tf.nn.conv2d(real_im, w_sum, strides=[1,1,1,1], padding='VALID')

                out_im_var = tf.nn.conv2d(out_im_sq, w_sum, strides=[1,1,1,1], padding='VALID') - out_im_mean**2
                real_im_var = tf.nn.conv2d(real_im_sq, w_sum, strides=[1,1,1,1], padding='VALID') - real_im_mean**2

                sq_var_diff = tf.reduce_mean((out_im_var - real_im_var)**2, axis=0)
                sq_var_diff = tf.reduce_sum(sq_var_diff)

                var_diff += sq_var_diff
            sq_var_diff = var_diff

        if 'pixel_wise_l2' in model.opts:
            if model.opts['pixel_wise_l2'] is True:
                # pixelwise l2
                sq_mean_diff = tf.reduce_mean((out_im - real_im)**2, axis=0)
                sq_mean_diff = tf.reduce_sum(sq_mean_diff)
        else:
            # l2 over mean of patch
            sq_mean_diff = tf.reduce_mean((out_im_mean - real_im_mean)**2, axis=0)
            sq_mean_diff = tf.reduce_sum(sq_mean_diff)



        if 'adv_cost_lambda' in model.opts:
            patch_var_lambda = model.opts['adv_cost_lambda']
        else:
            patch_var_lambda = 1.0

        model.loss_reconstruction = tf.add(sq_mean_diff, patch_var_lambda * sq_var_diff, name='loss_reconstruction')

    all_losses.append(model.loss_reconstruction)

    if model.opts['loss_regulariser'] in ['VAE', 'beta_VAE']:
        if model.opts['loss_regulariser'] == 'VAE':
            beta = 1
        else:
            beta = model.opts['beta']
        KL_divergence = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.exp(model.z_logvar) - model.z_logvar + model.z_mean**2,axis=1) - model.z_dim)
        model.loss_regulariser = tf.multiply(KL_divergence, beta, name="loss_regulariser")

    elif model.opts['loss_regulariser'] == 'WAE_MMD':
        mmds = []
        for C in model.opts['IMQ_length_params']:
            mmds.append(_mmd_init(model, C))
        #TODO: double check correct
        model.loss_regulariser = tf.multiply(tf.add_n(mmds),
                                             model.opts['lambda_imq'],
                                             name="loss_regulariser")
    elif model.opts['loss_regulariser'] is None:
        model.loss_regulariser = tf.constant(0, dtype=tf.float32, name="loss_regulariser")

    all_losses.append(model.loss_regulariser)

    if model.opts['z_logvar_regularisation'] == 'L1':
        model.z_logvar_loss = model.opts['lambda_logvar_regularisation'] * tf.reduce_mean(tf.reduce_sum(tf.abs(model.z_logvar), axis=1), name="z_logvar_loss")
        all_losses.append(model.z_logvar_loss)
    elif model.opts['z_logvar_regularisation'] == 'L2_squared':
        model.z_logvar_loss = model.opts['lambda_logvar_regularisation'] * tf.reduce_mean(tf.reduce_sum(tf.square(model.z_logvar), axis=1), name="z_logvar_loss")
        all_losses.append(model.z_logvar_loss)
    model.loss_total = tf.add_n(all_losses)

def optimizer_init(model):
    encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    if model.opts['optimizer'] == 'adam':
        model.learning_rate = tf.placeholder(tf.float32)
        model.train_step = tf.train.AdamOptimizer(model.learning_rate).minimize(model.loss_total, var_list=encoder_vars+decoder_vars)

        if model.opts['loss_reconstruction'] in ['L2_squared+adversarial', 'L2_squared+adversarial+l2_filter', 'L2_squared+multilayer_conv_adv', 'L2_squared+adversarial+l2_norm']:
            adv_cost_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='adversarial_cost')
            model.adv_cost_train_step = tf.train.AdamOptimizer(model.learning_rate).minimize(-model.adv_cost_loss, var_list=adv_cost_vars)

def _mmd_init(model, C):
    batch_size = tf.shape(model.input)[0]
    batch_size_float = tf.cast(batch_size, tf.float32)


    # ===== Inverse multiquadratic kernel MMD ============

    # 2*z_dim is recommended in the original WAE paper
    C_const = tf.cast(tf.constant(C), tf.float32)

    prior_tensor_ax0_rep = tf.tile(model.z_prior_sample[None, :, :], [batch_size, 1, 1])
    prior_tensor_ax1_rep = tf.tile(model.z_prior_sample[:, None, :], [1, batch_size, 1])
    q_tensor_ax0_rep = tf.tile(model.z_sample[None, :, :], [batch_size, 1, 1])
    q_tensor_ax1_rep = tf.tile(model.z_sample[:, None, :], [1, batch_size, 1])

    # prior_tensor_ax0_rep[a,b] = z_prior_sample[b]
    # prior_tensor_ax1_rep[a,b] = z_prior_sample[a]
    # prior_tensor_ax0_rep[a,b] - prior_tensor_ax1_rep[a,b] = z_prior_sample[b] - z_prior_sample[a]

    k_pp = C_const / (C_const + tf.reduce_sum((prior_tensor_ax0_rep - prior_tensor_ax1_rep) ** 2, axis=2))
    # k_pp[a, b] = C / (C + || z_prior_sample[b] - z_prior_sample[a] ||_L2^2)

    k_qq = C_const / (C_const + tf.reduce_sum((q_tensor_ax0_rep - q_tensor_ax1_rep) ** 2, axis=2))
    # k_pp[a, b] = C / (C + || z_sample[b] - z_sample[a] ||_L2^2)

    k_pq = C_const / (C_const + tf.reduce_sum((q_tensor_ax0_rep - prior_tensor_ax1_rep) ** 2, axis=2))
    # k_pq[a, b] = C / (C + || z_sample[b] - z_prior_sample[a] ||_L2^2)

    MMD_IMQ = (tf.reduce_sum(k_pp) - tf.reduce_sum(tf.diag_part(k_pp)) +
                                tf.reduce_sum(k_qq) - tf.reduce_sum(tf.diag_part(k_qq)) -
                                2 * (tf.reduce_sum(k_pq) - tf.reduce_sum(tf.diag_part(k_pq)))) / \
                   (batch_size_float * (batch_size_float - 1))
    return MMD_IMQ

def _dcgan_encoder(model):
    x = model.input
    model.x_flattened = tf.reshape(x, shape=[-1, np.prod(model.data_dims)])
    # adapted from Ilya's dcgan_encoder in https://github.com/tolstikhin/wae/blob/master/models.py
    num_units = model.opts['encoder_num_filters']
    num_layers = model.opts['encoder_num_layers']
    conv_filter_dim = model.opts['conv_filter_dim']
    layer_x = model.input
    for i in range(num_layers):
        scale = 2**(num_layers - i - 1)
        layer_x = tf.layers.conv2d(layer_x,
                                   filters=(num_units // scale),
                                   strides=2,
                                   kernel_size=[conv_filter_dim, conv_filter_dim],
                                   padding='same')
        layer_x = tf.layers.batch_normalization(layer_x)
        layer_x = tf.nn.relu(layer_x)

    dim = (model.data_dims[0] // (2 ** num_layers)) * (model.data_dims[1] // (2 ** num_layers)) * num_units
    layer_x = tf.reshape(layer_x, shape=[-1, dim])
    if model.opts['z_mean_activation'] == 'tanh':
        model.z_mean = tf.layers.dense(inputs=layer_x,
                                       units=model.z_dim,
                                       activation=tf.nn.tanh,
                                       name="z_mean")
    elif model.opts['z_mean_activation'] is None:
        model.z_mean = tf.layers.dense(inputs=layer_x,
                                       units=model.z_dim,
                                       name="z_mean")

    if model.opts['encoder_distribution'] != 'deterministic':
        model.z_logvar = tf.layers.dense(inputs=layer_x,
                                         units=model.z_dim,
                                         name="z_logvar")


def _dcgan_decoder(model):
    # adapted from Ilya's _dcgan_decoder in https://github.com/tolstikhin/wae/blob/master/models.py
    output_shape = model.data_dims
    num_units = model.opts['decoder_num_filters']
    batch_size = model.batch_size
    num_layers = model.opts['decoder_num_layers']
    conv_filter_dim = model.opts['conv_filter_dim']
    if model.opts['decoder_architecture'] == 'dcgan':
        height = output_shape[0] // 2**num_layers
        width = output_shape[1] // 2**num_layers
    elif model.opts['decoder_architecture'] == 'dcgan_mod':
        height = output_shape[0] // 2**(num_layers - 1)
        width = output_shape[1] // 2**(num_layers - 1)

    h0 = tf.layers.dense(inputs=model.z_sample,
                         units=num_units * height * width)
    h0 = tf.reshape(h0, [-1, height, width, num_units])
    h0 = tf.nn.relu(h0)
    layer_x = h0
    for i in range(num_layers - 1):
        scale = 2**(i + 1)
        _out_shape = [batch_size, height * scale,
                      width * scale, num_units // scale]
        _in_shape = layer_x.get_shape().as_list()
        assert len(_in_shape) == 4, 'Conv2d_transpose works only with 4d tensors.'

        layer_x = tf.layers.conv2d_transpose(inputs=layer_x,
                                             filters=_out_shape[-1],
                                             kernel_size=conv_filter_dim,
                                             strides=2,
                                             padding='same')
        layer_x = tf.layers.batch_normalization(layer_x)
        layer_x = tf.nn.relu(layer_x)
    _out_shape = [batch_size] + list(output_shape)
    if model.opts['decoder_architecture'] == 'dcgan':
        model.x_logits_img_shape = tf.layers.conv2d_transpose(inputs=layer_x,
                                                              filters=_out_shape[-1],
                                                              kernel_size=conv_filter_dim,
                                                              strides=2,
                                                              padding='same',
                                                              name='x_logits')
        model.x_logits = tf.reshape(model.x_logits_img_shape,
                                    shape=[-1, np.prod(model.data_dims)],
                                    name="x_logits")
    elif model.opts['decoder_architecture'] == 'dcgan_mod':
        model.x_logits_img_shape = tf.layers.conv2d_transpose(inputs=layer_x,
                                                              filters=_out_shape[-1],
                                                              kernel_size=conv_filter_dim,
                                                              strides=1,
                                                              padding='same',
                                                              name='x_logits')
        model.x_logits = tf.reshape(model.x_logits_img_shape,
                                    shape=[-1, np.prod(model.data_dims)],
                                    name="x_logits")


def _encoder_small_convolutional_celebA_init(model):
    '''
    returns [z_mean, z_logvar]
    z_mean:  mean of encoding distributions
    z_logvar: the log-variance or scale of axis aligned spread
        For Gaussian encoder, this is the log-variance of spread in each dimension
        For uniform encoder, this is side length of uniform box in each dimension
        For deterministic encoder, this is None
    '''
    x = model.input
    model.x_flattened = tf.reshape(x, shape=[-1, np.prod(model.data_dims)])

    Q_conv1 = tf.layers.batch_normalization(
                tf.layers.conv2d(inputs=x,
                                 filters=128,
                                 kernel_size=[3, 3],
                                 padding="same",
                                 activation=tf.nn.elu))

    Q_pool1 = tf.layers.max_pooling2d(inputs=Q_conv1,
                                      pool_size=[2, 2],
                                      strides=2)

    Q_conv2 = tf.layers.batch_normalization(
                tf.layers.conv2d(inputs=Q_pool1,
                                 filters=256,
                                 kernel_size=[3, 3],
                                 padding="same",
                                 activation=tf.nn.elu))

    Q_pool2 = tf.layers.max_pooling2d(inputs=Q_conv2,
                                      pool_size=[2, 2],
                                      strides=2)

    Q_conv3 = tf.layers.batch_normalization(
                tf.layers.conv2d(inputs=Q_pool2,
                                 filters=256,
                                 kernel_size=[3, 3],
                                 padding="same",
                                 activation=tf.nn.elu))

    Q_pool3 = tf.layers.max_pooling2d(inputs=Q_conv3,
                                      pool_size=[2, 2],
                                      strides=2)

    Q_conv4 = tf.layers.batch_normalization(
                tf.layers.conv2d(inputs=Q_pool3,
                                 filters=256,
                                 kernel_size=[3, 3],
                                 padding="same",
                                 activation=tf.nn.elu))

    Q_pool4 = tf.layers.max_pooling2d(inputs=Q_conv4,
                                      pool_size=[2, 2],
                                      strides=2)

    Q_conv5 = tf.layers.batch_normalization(
                tf.layers.conv2d(inputs=Q_pool4,
                                 filters=256,
                                 kernel_size=[3, 3],
                                 padding="same",
                                 activation=tf.nn.elu))

    Q_pool5 = tf.layers.max_pooling2d(inputs=Q_conv5,
                                      pool_size=[2, 2],
                                      strides=2)


    Q_pool5_flat = tf.reshape(Q_pool5, shape=[-1, 2*2*256])

    if model.opts['z_mean_activation'] == 'tanh':
        model.z_mean = tf.layers.dense(inputs=Q_pool5_flat,
                                       units=model.z_dim,
                                       activation=tf.nn.tanh,
                                       name="z_mean")
    elif model.opts['z_mean_activation'] is None:
        model.z_mean = tf.layers.dense(inputs=Q_pool5_flat,
                                       units=model.z_dim,
                                       name="z_mean")

    if model.opts['encoder_distribution'] == 'deterministic':
        return model.z_mean, None
    else:
        model.z_logvar = tf.layers.dense(inputs=Q_pool5_flat,
                                         units=model.z_dim,
                                         name="z_logvar")
        return model.z_mean, model.z_logvar

def _encoder_FC_dsprites_init(model):
    if len(model.input.shape) == 3: # then we have to explicitly add single channel at end
        x_reshape = tf.reshape(model.input,
                               shape=(-1,) + model.train_data.shape[1:] + (1,))
    else:
        x_reshape = model.input

    model.x_flattened = tf.reshape(model.input, shape=[-1, np.prod(model.data_dims)])

    Q_FC1 = tf.layers.dense(inputs=model.x_flattened,
                            units=1200,
                            activation=tf.nn.relu)
    Q_FC2 = tf.layers.dense(inputs=Q_FC1,
                            units=1200,
                            activation=tf.nn.relu)

    if model.opts['z_mean_activation'] == 'tanh':
        model.z_mean = tf.layers.dense(inputs=Q_FC2, units=model.z_dim, activation=tf.nn.tanh, name="z_mean")
    elif model.opts['z_mean_activation'] is None:
        model.z_mean = tf.layers.dense(inputs=Q_FC2, units=model.z_dim, name="z_mean")

    if model.opts['encoder_distribution'] == 'deterministic':
        return model.z_mean, None
    else:
        model.z_logvar = tf.layers.dense(inputs=Q_FC2, units=model.z_dim, name="z_logvar")
        return model.z_mean, model.z_logvar

def _z_sample_init(model):
    '''
    return z_sample: one sample from the encoding distribution
    '''
    if model.opts['encoder_distribution'] == 'deterministic':
        model.z_sample = tf.add(model.z_mean, 0, name="z_sample")
        return model.z_sample
    else:
        if model.opts['logvar-clipping'] is not None:
            # clipping of logvariances to prevent numerical errors
            clip_lower, clip_upper = model.opts['logvar-clipping']
            model.z_logvar = tf.clip_by_value(model.z_logvar, clip_lower, clip_upper)

        if model.opts['encoder_distribution'] == 'gaussian':
            eps = tf.random_normal(shape=tf.shape(model.z_mean))
            noise = tf.exp(model.z_logvar  / 2) * eps
        elif model.opts['encoder_distribution'] == 'uniform':
            eps = tf.random_uniform(shape=tf.shape(model.z_mean))
            noise = tf.exp(model.z_logvar) * eps
        model.z_sample = tf.add(model.z_mean, noise, name="z_sample")
    return model.z_sample

def _decoder_small_convolutional_celebA_init(model):
    P_dense1 = tf.layers.dense(inputs=model.z_sample,
                               units=256*4*4,
                               activation=tf.nn.elu)
    P_dense1_reshape = tf.reshape(P_dense1, shape=[-1, 4, 4, 256])

    P_conv1 = tf.nn.elu(
                tf.layers.batch_normalization(
                    tf.layers.conv2d(inputs=P_dense1_reshape,
                                     filters=256,
                                     kernel_size=[3, 3],
                                     padding="same")))

    P_conv2 = tf.nn.elu(
                tf.layers.batch_normalization(
                    tf.layers.conv2d(inputs=P_conv1,
                                     filters=128,
                                     kernel_size=[3, 3],
                                     padding="same")))


    P_conv2_upsampled = tf.image.resize_nearest_neighbor(images=P_conv2, size=[8, 8])

    P_conv3 = tf.nn.elu(
                tf.layers.batch_normalization(
                    tf.layers.conv2d(inputs=P_conv2_upsampled,
                                     filters=64,
                                     kernel_size=[3, 3],
                                     padding="same")))

    P_conv4 = tf.nn.elu(
                tf.layers.batch_normalization(
                    tf.layers.conv2d(inputs=P_conv3,
                                     filters=64,
                                     kernel_size=[3, 3],
                                     padding="same")))

    P_conv4_upsampled = tf.image.resize_nearest_neighbor(images=P_conv4,
                                                         size=[16, 16])
    P_conv5 = tf.nn.elu(
                tf.layers.batch_normalization(
                    tf.layers.conv2d(inputs=P_conv4_upsampled,
                                     filters=32,
                                     kernel_size=[3, 3],
                                     padding="same")))

    P_conv6 = tf.nn.elu(
                tf.layers.batch_normalization(
                    tf.layers.conv2d(inputs=P_conv5,
                                     filters=32,
                                     kernel_size=[3, 3],
                                     padding="same")))


    P_conv6_upsampled = tf.image.resize_nearest_neighbor(images=P_conv6,
                                                         size=[32, 32])

    P_conv7 = tf.nn.elu(
                tf.layers.batch_normalization(
                    tf.layers.conv2d(inputs=P_conv6_upsampled,
                                     filters=16,
                                     kernel_size=[3, 3],
                                     padding="same")))

    P_conv8 = tf.nn.elu(
                tf.layers.batch_normalization(
                    tf.layers.conv2d(inputs=P_conv7,
                                     filters=16,
                                     kernel_size=[3, 3],
                                     padding="same")))


    P_conv8_upsampled = tf.image.resize_nearest_neighbor(images=P_conv8,
                                                         size=[64, 64])


    model.x_logits_img_shape = tf.layers.conv2d(inputs=P_conv8_upsampled,
                                                filters=3,
                                                kernel_size=[3, 3],
                                                padding="same",
                                                name="x_logits_img_shape")

    model.x_logits = tf.reshape(model.x_logits_img_shape,
                                shape=[-1, np.prod(model.data_dims)],
                                name="x_logits")

def _decoder_FC_dsprites_init(model):
    P_FC1 = tf.layers.dense(inputs=model.z_sample, units=1200, activation=tf.nn.tanh)
    P_FC2 = tf.layers.dense(inputs=P_FC1, units=1200, activation=tf.nn.tanh)
    P_FC3 = tf.layers.dense(inputs=P_FC2, units=1200, activation=tf.nn.tanh)
    if model.data_dims[0] == 64:
        model.x_logits = tf.layers.dense(inputs=P_FC3, units=4096, name="x_logits")
    elif model.data_dims[0] == 32:
        model.x_logits = tf.layers.dense(inputs=P_FC3, units=1024, name="x_logits")

    model.x_logits_img_shape = tf.reshape(model.x_logits,[-1, model.data_dims[0], model.data_dims[1], 1],
                                          name="x_logits_img_shape")

def lrelu(alpha, inputs):
    return tf.maximum(inputs, alpha*inputs)

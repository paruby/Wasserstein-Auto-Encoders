import tensorflow as tf
import numpy as np
import os
import time
import sys
import models

class Model(object):
    def __init__(self, opts, ...):
        self.sess = tf.Session()

        self.opts = opts
        self.z_dim = self.opts['z_dim']
        self.batch_size = self.opts['batch_size']
        self.train_data, self.test_data = utils.load_data(self, seed=0)#TODO
        
        self.fixed_test_sample = self.sample_minibatch(test=True, seed=0)
        self.fixed_train_sample = self.sample_minibatch(test=False, seed=0)
        self.fixed_codes = self.sample_codes(seed=0)

        self.data_dims = self.train_data.shape[1:]
        self.input = self.input = tf.placeholder(tf.float32, (None,) + self.data_dims, name="input")

        self.losses_train = []
        self.losses_test_random = []
        self.losses_test_fixed = []




        models.encoder_init(self)
        models.decoder_init(self)
        models.pror_init(self)
        models.loss_init(self)
        models.optimizer_init(self)
        if self.opts['make_pictures_every'] is not None:
            utils.plot_all_init(self)

        self.sess.run(tf.global_variables_initializer())


    def train(self):
        if self.opts['optimizer'] == 'adam':
            learning_rates = [i[0] for i in self.opts['learning_rate_schedule']]
            iterations_list = [i[1] for i in self.opts['learning_rate_schedule']]
            total_num_iterations = iterations_list[-1]
            it = 0
            while it < total_num_iterations:
                if it > lr_iterations:
                    lr_counter += 1
                    lr = learning_rates[lr_counter]
                    lr_iterations = iterations_list[lr_counter]

                self.sess.run(
                    self.train_step,
                    feed_dict={self.learning_rate: learning_rate,
                               self.input: self.sample_data(self.batch_size)}
                    )

                if (self.opts['print_log_information'] is True) and (it % 100 == 0):
                    utils.print_log_information(self, it)

                if self.opts['make_pictures_every'] is not None:
                    if (it > 0) and (it % self.opts['make_pictures_every'] == 0):
                        utils.plot_all(self, it)


    def encode(self, images, mean=True):
        if mean is False:
            return self.sess.run(self.z_sample, feed_dict={self.input: images})
        if mean is True:
            return self.sess.run(self.z_mean, feed_dict={self.input: images})


    def decode(self, codes):
        return self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape), feed_dict={self.z_sample: codes})

    def sample_codes(self, batch_size=None, seed=None):
        if batch_size is None:
            batch_size = self.batch_size
        if seed is not None:
            st0 = np.random.get_state()
            np.random.seed(seed)
        # z_mean here is just a placeholder so that z_prior_sample knows what size to be
        codes = self.sess.run(self.z_prior_sample, feed_dict={self.z_mean: np.random.randn(batch_size, self.z_dim)})
        if seed is not None:
            np.random.set_state(st0)
        return codes

    def sample_minibatch(self, batch_size=None, test=False, seed=None):
        if seed is not None:
            st0 = np.random.get_state()
            np.random.seed(seed)
        if batch_size is None:
            batch_size = self.batch_size
        if test is False:
            sample = self.train_data[np.random.choice(range(len(self.train_data)), batch_size, replace=False)]
        else:
            sample = self.test_data[np.random.choice(range(len(self.test_data)), batch_size, replace=False)]
        if seed is not None:
            np.random.set_state(st0)
        return sample

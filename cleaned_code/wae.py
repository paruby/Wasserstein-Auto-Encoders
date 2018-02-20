import tensorflow as tf
import numpy as np
import os
import time
import sys
import models
import config
import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import disentanglement_metric

class Model(object):
    def __init__(self, opts):
        self.sess = tf.Session()

        self.opts = opts
        utils.opts_check(self)

        self.z_dim = self.opts['z_dim']
        self.batch_size = self.opts['batch_size']
        self.train_data, self.test_data = utils.load_data(self, seed=0)

        self.data_dims = self.train_data.shape[1:]
        self.input = tf.placeholder(tf.float32, (None,) + self.data_dims, name="input")

        self.losses_train = []
        self.losses_test_random = []
        self.losses_test_fixed = []

        self.experiment_path = self.opts['experiment_path']
        utils.create_directories(self)
        utils.save_opts(self)
        utils.copy_all_code(self)

        models.encoder_init(self)
        models.decoder_init(self)
        models.prior_init(self)
        models.loss_init(self)
        models.optimizer_init(self)

        self.fixed_test_sample = self.sample_minibatch(test=True, seed=0)
        self.fixed_train_sample = self.sample_minibatch(test=False, seed=0)
        self.fixed_codes = self.sample_codes(seed=0)


        if self.opts['make_pictures_every'] is not None:
            utils.plot_all_init(self)

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
        self.sess.run(tf.global_variables_initializer())

    def save(self, it):
        model_path = "checkpoints/"
        save_path = self.saver.save(self.sess, model_path, global_step=it)
        print("Model saved to: %s" % save_path)

    def restore(self, model_path):
        self.saver.restore(self.sess, model_path)
        print("Model restored from : %s" % model_path)

    def train(self):
        print("Beginning training")
        if self.opts['optimizer'] == 'adam':
            learning_rates = [i[0] for i in self.opts['learning_rate_schedule']]
            iterations_list = [i[1] for i in self.opts['learning_rate_schedule']]
            total_num_iterations = iterations_list[-1]
            it = 0
            lr_counter = 0
            lr = learning_rates[lr_counter]
            lr_iterations = iterations_list[lr_counter]
            while it < total_num_iterations:
                if it % 1000 == 0:
                    print("\nIteration %i" % it, flush=True)
                if it % 100 == 0:
                    print('.', end='', flush=True)
                it += 1
                if it > lr_iterations:
                    lr_counter += 1
                    lr = learning_rates[lr_counter]
                    lr_iterations = iterations_list[lr_counter]

                self.sess.run(
                    self.train_step,
                    feed_dict={self.learning_rate: lr,
                               self.input: self.sample_minibatch(self.batch_size)}
                    )

                if (self.opts['print_log_information'] is True) and (it % 100 == 0):
                    utils.print_log_information(self, it)

                if self.opts['make_pictures_every'] is not None:
                    if it % self.opts['make_pictures_every'] == 0:
                        utils.plot_all(self, it)

                if it % self.opts['save_every'] == 0:
                    self.save(it)
        # once training is complete, calculate disentanglement metric
        if 'disentanglement_metric' in self.opts:
            if self.opts['disentanglement_metric'] is True:
                self.disentanglement = disentanglement_metric.Disentanglement(self)
                self.disentanglement.do_all(it)

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

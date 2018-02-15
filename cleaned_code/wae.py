import tensorflow as tf
import numpy as np
import os
import time
import sys
import models

class Model(object):
    def __init__(self, opts, ...):
        self.test_data =
        self.train_data =
        self.data_dims = self.train_data.shape[1:]
        self.input = self.input = tf.placeholder(tf.float32, (None,) + self.data_dims, name="input")

        self.mini_batch_size = BATCH_SIZE



        self.encoder = models.encoder_init(self)
        self.decoder = models.decoder_init(self)
        self.prior = models.pror_init(self)
        self.loss = models.loss_init(self)
        self.optimizer = models.optimizer_init(self)

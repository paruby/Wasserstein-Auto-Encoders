import numpy as np
import tensorflow as tf
import utils

class Disentanglement(object):
    def __init__(self, model):
        self.model = model
        self.imgs, self.latents_sizes, self.latents_bases = utils.load_disentanglement_data_dsprites()
        self._classifier_init()
        self._classifier_init5()

    def do_all(self, iteration_num):
        print("Calculating disentanglement metric on 4 variables")
        print("Generating data")
        Z_diff, Y = self.generate_data(30, 5000) # they use 5000 batches
        print("\nTraining classifier")
        self.train_classifier(Z_diff, Y, iteration_num)

        print("Calculating disentanglement metric on 5 variables")
        print("Generating data")
        Z_diff, Y = self.generate_data(30, 5000, 5) # they use 5000 batches
        print("\nTraining classifier")
        self.train_classifier5(Z_diff, Y, iteration_num)

    def train_classifier(self, Z_diff, Y, iteration_num):
        with open("disentanglement4.txt", "a") as disentanglement_file:
            disentanglement_file.write("\nIteration %i" % iteration_num)
        for _ in range(3):
            print('.', end='', flush=True)
            self.model.sess.run(tf.variables_initializer(self.classifier_vars + self.optimiser_vars))
            for i in range(100000):
                self.model.sess.run(self.train_step, feed_dict={self.classifier_input: Z_diff, self.true_labels:Y})
            final_accuracy = self.model.sess.run(self.accuracy, feed_dict={self.classifier_input: Z_diff, self.true_labels:Y})
            with open("disentanglement4.txt", "a") as disentanglement_file:
                disentanglement_file.write("\n%g" % final_accuracy)
        print("\n")
        return

    def train_classifier5(self, Z_diff, Y, iteration_num):
        with open("disentanglement5.txt", "a") as disentanglement_file:
            disentanglement_file.write("\nIteration %i" % iteration_num)
        for _ in range(3):
            self.model.sess.run(tf.variables_initializer(self.classifier_vars5 + self.optimiser_vars5))
            for i in range(100000):
                self.model.sess.run(self.train_step5, feed_dict={self.classifier_input5: Z_diff, self.true_labels5:Y})
            final_accuracy = self.model.sess.run(self.accuracy5, feed_dict={self.classifier_input5: Z_diff, self.true_labels5:Y})
            with open("disentanglement5.txt", "a") as disentanglement_file:
                disentanglement_file.write("\n%g" % final_accuracy)
        return

    def generate_data(self, L, B, num_features=4):
        Y = []
        Z_diff = []
        for b in range(B):
            if b % 500 == 0:
                print('.', end='', flush=True)
            y = np.random.randint(1,num_features+1)
            Y.append(y-1)  # for convenience when implementing the classifier, subtract 1 to get labels 0,...,3
            v1 = self.sample_latent(L)
            v2 = self.sample_latent(L)
            # -y because posX, posY, scale and rotation are last 4 latent factors
            v2[:,-y] = v1[:,-y]
            im1 = self.imgs[self.latent_to_index(v1)][:,:,:,None]
            im2 = self.imgs[self.latent_to_index(v2)][:,:,:,None]
            # need to be n_images x height x width x channels
            z1 = self.model.encode(im1)
            z2 = self.model.encode(im2)
            z_diff_L = np.abs(z1 - z2)
            Z_diff.append(np.mean(z_diff_L, axis=0))
        Z_diff = np.array(Z_diff)
        Y = np.array(Y)
        return Z_diff, Y



    def _classifier_init(self):
        self.classifier_input = tf.placeholder(tf.float32, shape=[None, self.model.z_dim])
        self.true_labels = tf.placeholder(tf.int32, shape=[None])
        onehot = tf.one_hot(self.true_labels, depth=4)
        with tf.variable_scope("classifier"):
            self.evidence = tf.layers.dense(inputs=self.classifier_input, units=4)
        self.predictions = tf.argmax(self.evidence, axis=1, output_type=tf.int32)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.evidence, labels=onehot))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.true_labels), tf.float32))

        self.classifier_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classifier')
        with tf.variable_scope("optimiser"):
            self.train_step = tf.train.AdagradOptimizer(1e-2).minimize(self.loss, var_list=self.classifier_vars)
        self.optimiser_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='optimiser')

    def _classifier_init5(self):
        self.classifier_input5 = tf.placeholder(tf.float32, shape=[None, self.model.z_dim])
        self.true_labels5 = tf.placeholder(tf.int32, shape=[None])
        onehot = tf.one_hot(self.true_labels5, depth=5)
        with tf.variable_scope("classifier5"):
            self.evidence5 = tf.layers.dense(inputs=self.classifier_input5, units=5)
        self.predictions5 = tf.argmax(self.evidence5, axis=1, output_type=tf.int32)
        self.loss5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.evidence5, labels=onehot))
        self.accuracy5 = tf.reduce_mean(tf.cast(tf.equal(self.predictions5, self.true_labels5), tf.float32))

        self.classifier_vars5 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='classifier5')
        with tf.variable_scope("optimiser5"):
            self.train_step5 = tf.train.AdagradOptimizer(1e-2).minimize(self.loss5, var_list=self.classifier_vars5)
        self.optimiser_vars5 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='optimiser5')

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
        return samples

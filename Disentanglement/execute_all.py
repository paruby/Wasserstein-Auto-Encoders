readme = """
Purpose of running this experiment:

I am going to implement the disentanglement metric proposed by the beta-VAE paper

"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import os
import time
import sys
from disentanglement_metric import Disentanglement


class WAE_MMD(object):
    def __init__(self, train_data, test_data, data_in_order, LATENT_DIMENSIONALITY, LAMBDA_IMQ, LAMBDA_L1, BATCH_SIZE):
        self.K = LATENT_DIMENSIONALITY
        self.LAMBDA_IMQ = LAMBDA_IMQ
        self.LAMBDA_L1 = LAMBDA_L1

        self.train_data = train_data
        self.test_data = test_data

        self.data_dims = self.train_data.shape[1:]

        self.data_in_order = data_in_order
        self.mini_batch_size = BATCH_SIZE

        # Define model
        self.input = tf.placeholder(tf.float32, (None,) + self.data_dims, name="input")

        self._encoder_init()
        self._decoder_init()
        self._prior_init()
        self._optimizer_init()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self._plot_all_init()

        self.objective_losses = []
        self.test_losses = []


        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
        self.disentanglement = Disentanglement(self)

    def plot_all(self, filename_appendage):
        fig = plt.figure(figsize=(50, 50))
        fig.suptitle("Cols: {Test, Train, Prior samples}.  Rows: {Reconstruction, Interpolations, Axis walks}", fontsize=40, fontweight='bold')
        outer = gridspec.GridSpec(4, 3, wspace=0.2, hspace=0.2)
        # outer[0] is reconstructions of train images
        # outer[1] is reconstructions of test images
        # outer[2] is random samples

        # outer[3] is spherical interpolations between embeddings of train images
        # outer[4] is spherical interpolations between embeddings of test images
        # outer[5] is spherical interpolations between embeddings of latent samples

        # outer[6] is axis walks starting at embeddings of train images
        # outer[7] is axis walks starting at embeddings of test images
        # outer[8] is axis walks starting at random latent samples

        # outer[9] is test and train reconstruction errors
        # outer[10] is ..... do we need some representation of logvars and MMD?
        # outer[11] is the differnece between distributions plot that I need code up still

        # ========== Reconstructions of real images ===========
        # Plot reconstructions of training images

        inner = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=outer[0], wspace=0.1, hspace=0.1)

        train_reconstruction_images = self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape),
                                             feed_dict={self.input: self.train_reconstruction_images})

        for j in range(0, 60, 2):
            ax_real = plt.Subplot(fig, inner[j])
            ax_real.imshow(self.train_reconstruction_images[j//2, :, :], cmap="gray")
            ax_real.axis("off")
            fig.add_subplot(ax_real)

            ax_recon = plt.Subplot(fig, inner[j + 1])
            ax_recon.imshow(train_reconstruction_images[j//2, :, :, 0], cmap="gray")
            ax_recon.axis("off")
            fig.add_subplot(ax_recon)

        sample_train_images = self.sample_data(batch_size=20)
        sample_reconstructed_images = self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape),
                                             feed_dict={self.input: sample_train_images})

        for j in range(60, 100, 2):
            ax_real = plt.Subplot(fig, inner[j])
            ax_real.imshow(sample_train_images[j//2 - 30, :, :], cmap="gray")
            ax_real.axis("off")
            fig.add_subplot(ax_real)

            ax_recon = plt.Subplot(fig, inner[j + 1])
            ax_recon.imshow(sample_reconstructed_images[j//2 - 30, :, :, 0], cmap="gray")
            ax_recon.axis("off")
            fig.add_subplot(ax_recon)






        # Plot reconstructions of test images
        inner = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=outer[1], wspace=0.1, hspace=0.1)

        test_reconstruction_images = self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape),
                                             feed_dict={self.input: self.test_reconstruction_images})

        for j in range(0, 60, 2):
            ax_real = plt.Subplot(fig, inner[j])
            ax_real.imshow(self.test_reconstruction_images[j//2, :, :], cmap="gray")
            ax_real.axis("off")
            fig.add_subplot(ax_real)

            ax_recon = plt.Subplot(fig, inner[j + 1])
            ax_recon.imshow(test_reconstruction_images[j//2, :, :, 0], cmap="gray")
            ax_recon.axis("off")
            fig.add_subplot(ax_recon)

        sample_test_images = self.sample_data(batch_size=20, test=True)
        sample_reconstructed_images = self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape),
                                             feed_dict={self.input: sample_test_images})

        for j in range(60, 100, 2):
            ax_real = plt.Subplot(fig, inner[j])
            ax_real.imshow(sample_test_images[j//2 - 30, :, :], cmap="gray")
            ax_real.axis("off")
            fig.add_subplot(ax_real)

            ax_recon = plt.Subplot(fig, inner[j + 1])
            ax_recon.imshow(sample_reconstructed_images[j//2 - 30, :, :, 0], cmap="gray")
            ax_recon.axis("off")
            fig.add_subplot(ax_recon)



        # Plot images generated from fixed random samples
        random_sample_images = self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape),
                                             feed_dict={self.z_sample: self.random_codes})

        inner = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=outer[2], wspace=0.1, hspace=0.1)

        fixed_random_sample_images = self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape),
                                             feed_dict={self.z_sample: self.random_codes})

        for j in range(0, 60, 1):
            ax = plt.Subplot(fig, inner[j])
            ax.imshow(fixed_random_sample_images[j, :, :, 0], cmap="gray")
            ax.axis("off")
            fig.add_subplot(ax)


        random_codes = self.sess.run(self.z_prior_sample, feed_dict={self.z_sample: np.random.randn(40, self.K)})
        varying_random_sample_images = self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape), feed_dict={self.z_sample: random_codes})

        for j in range(60, 100, 1):
            ax = plt.Subplot(fig, inner[j])
            ax.imshow(varying_random_sample_images[j - 60, :, :, 0], cmap="gray")
            ax.axis("off")
            fig.add_subplot(ax)



        # ========== Interpolations ===========

        # Training data
        inner = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=outer[3], wspace=0.1, hspace=0.1)

        for i in range(5):
            m1 = self.sess.run(self.z_mean, feed_dict={self.input: self.train_data[i][None,:,:]})
            m2 = self.sess.run(self.z_mean, feed_dict={self.input: self.train_data[i+1][None,:,:]})
            embeddings = lerp(m1, m2, 10) # LINEAR for uniform box
            outputs = self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape),
                                    feed_dict={self.z_sample: embeddings})
            for j in range(10):
                ax = plt.Subplot(fig, inner[10*i + j])
                ax.imshow(outputs[j, :, :, 0], cmap='gray')
                ax.axis("off")
                fig.add_subplot(ax)

        for i in range(5, 10):
            m1 = self.sess.run(self.z_mean, feed_dict={self.input: self.sample_data(batch_size=1)})
            m2 = self.sess.run(self.z_mean, feed_dict={self.input: self.sample_data(batch_size=1)})
            embeddings = lerp(m1, m2, 10)
            outputs = self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape),
                                    feed_dict={self.z_sample: embeddings})
            for j in range(10):
                ax = plt.Subplot(fig, inner[10*i + j])
                ax.imshow(outputs[j, :, :, 0], cmap='gray')
                ax.axis("off")
                fig.add_subplot(ax)


        # Test data
        inner = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=outer[4], wspace=0.1, hspace=0.1)

        for i in range(5):
            m1 = self.sess.run(self.z_mean, feed_dict={self.input: self.test_data[i][None,:,:]})
            m2 = self.sess.run(self.z_mean, feed_dict={self.input: self.test_data[i+1][None,:,:]})
            embeddings = lerp(m1, m2, 10)
            outputs = self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape),
                                    feed_dict={self.z_sample: embeddings})
            for j in range(10):
                ax = plt.Subplot(fig, inner[10*i + j])
                ax.imshow(outputs[j, :, :, 0], cmap='gray')
                ax.axis("off")
                fig.add_subplot(ax)

        for i in range(5, 10):
            m1 = self.sess.run(self.z_mean, feed_dict={self.input: self.sample_data(batch_size=1, test=True)})
            m2 = self.sess.run(self.z_mean, feed_dict={self.input: self.sample_data(batch_size=1, test=True)})
            embeddings = lerp(m1, m2, 10)
            outputs = self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape),
                                    feed_dict={self.z_sample: embeddings})
            for j in range(10):
                ax = plt.Subplot(fig, inner[10*i + j])
                ax.imshow(outputs[j, :, :, 0], cmap='gray')
                ax.axis("off")
                fig.add_subplot(ax)




        # Random samples from the latent space
        inner = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=outer[5], wspace=0.1, hspace=0.1)

        for i in range(5):
            m1 = self.random_codes[i][None,:]
            m2 = self.random_codes[i+1][None,:]
            embeddings = lerp(m1, m2, 10)
            outputs = self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape),
                                    feed_dict={self.z_sample: embeddings})
            for j in range(10):
                ax = plt.Subplot(fig, inner[10*i + j])
                ax.imshow(outputs[j, :, :, 0], cmap='gray')
                ax.axis("off")
                fig.add_subplot(ax)

        for i in range(5, 10):
            m1 = self.sess.run(self.z_prior_sample, feed_dict={self.z_sample: np.random.randn(1, self.K)})
            m2 = self.sess.run(self.z_prior_sample, feed_dict={self.z_sample: np.random.randn(1, self.K)})
            embeddings = lerp(m1, m2, 10)
            outputs = self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape),
                                    feed_dict={self.z_sample: embeddings})
            for j in range(10):
                ax = plt.Subplot(fig, inner[10*i + j])
                ax.imshow(outputs[j, :, :, 0], cmap='gray')
                ax.axis("off")
                fig.add_subplot(ax)


        # ========== Walks along each axis direction ===========

        # Training data
        inner = gridspec.GridSpecFromSubplotSpec(self.K, 10, subplot_spec=outer[6], wspace=0.1, hspace=0.1)

        mean = self.sess.run(self.z_mean, feed_dict={self.input: self.sample_data(batch_size=1)})
        for axis in range(self.K):
            repeat_mean = np.repeat(mean, 10, axis=0)
            repeat_mean[:, axis] = np.linspace(-1,1,10)
            outputs = self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape),
                                    feed_dict={self.z_sample: repeat_mean})
            for j in range(10):
                ax = plt.Subplot(fig, inner[10*axis + j])
                ax.imshow(outputs[j, :, :, 0], cmap='gray')
                ax.axis("off")
                fig.add_subplot(ax)


        # Test data
        inner = gridspec.GridSpecFromSubplotSpec(self.K, 10, subplot_spec=outer[7], wspace=0.1, hspace=0.1)

        mean = self.sess.run(self.z_mean, feed_dict={self.input: self.sample_data(batch_size=1, test=True)})
        for axis in range(self.K):
            repeat_mean = np.repeat(mean, 10, axis=0)
            repeat_mean[:, axis] = np.linspace(-1,1,10)
            outputs = self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape),
                                    feed_dict={self.z_sample: repeat_mean})
            for j in range(10):
                ax = plt.Subplot(fig, inner[10*axis + j])
                ax.imshow(outputs[j, :, :, 0], cmap='gray')
                ax.axis("off")
                fig.add_subplot(ax)



        # Random samples from the latent space
        inner = gridspec.GridSpecFromSubplotSpec(self.K, 10, subplot_spec=outer[8], wspace=0.1, hspace=0.1)

        mean = self.sess.run(self.z_prior_sample, feed_dict={self.z_sample: np.random.randn(1, self.K)})
        for axis in range(self.K):
            repeat_mean = np.repeat(mean, 10, axis=0)
            repeat_mean[:, axis] = np.linspace(-1,1,10)
            outputs = self.sess.run(tf.nn.sigmoid(self.x_logits_img_shape),
                                    feed_dict={self.z_sample: repeat_mean})
            for j in range(10):
                ax = plt.Subplot(fig, inner[10*axis + j])
                ax.imshow(outputs[j, :, :, 0], cmap='gray')
                ax.axis("off")
                fig.add_subplot(ax)



        # ========== Training and test error plots ===========

        inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[9], wspace=0.1, hspace=0.1)

        ax = plt.Subplot(fig, inner[0])
        if len(self.objective_losses) > 5:
            ax.plot(np.log(self.objective_losses[3:]), linewidth=7.0)
            ax.plot(np.log(self.test_losses[3:]), linewidth=7.0)
        else:
            ax.plot(np.log(self.objective_losses), linewidth=7.0)
            ax.plot(np.log(self.test_losses), linewidth=7.0)
        ax.legend(["Log training loss", "Log test loss"], prop={'size': 40})
        fig.add_subplot(ax)


        # ========== Most non-gaussian 2d subspace ===========

        inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[10], wspace=0.1, hspace=0.1)

        real_embeddings = self.sess.run(self.z_sample, feed_dict={self.input: self.sample_data(batch_size=1000)})
        prior_samples = self.sess.run(self.z_prior_sample, feed_dict={self.z_sample: np.random.randn(1000, self.K)})

        real_projections, prior_projections = self.least_gaussian_2d_subspace(real_embeddings, prior_samples)

        ax = plt.Subplot(fig, inner[0])
        ax.scatter(real_projections[:,0], real_projections[:,1], s=50)
        ax.scatter(prior_projections[:,0], prior_projections[:,1], s=50)
        ax.legend(["Q(Z)", "P(Z)"], prop={'size': 40})
        fig.add_subplot(ax)





        fig.savefig("output/plots/" + filename_appendage + ".png")
        plt.close(fig)
        plt.close("all")

        return

    def least_gaussian_2d_subspace(self, real_embeddings, prior_samples):
        # adapted from ilya's https://github.com/tolstikhin/adagan/blob/master/pot.py#L1349
        real_embedding_var = self._proj_real_embedding_var
        prior_samples_var = self._proj_prior_samples_var
        optim = self._proj_optim
        loss = self._proj_loss
        v1 = self._proj_v1
        v2 = self._proj_v2
        #proj_mat = tf.concat([v, u], 1).eval()
        #dot_prod = -1
        best_of_runs = 10e5 # Any positive value would do
        updated = False

        for _start in range(3):
            # We will run 3 times from random inits
            loss_prev = 10e5 # Any positive value would do
            proj_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope="least_gaussian_2d_subspace")
            self.sess.run(tf.variables_initializer(proj_vars))
            step = 0
            for _ in range(5000):
                self.sess.run(optim, feed_dict={real_embedding_var:real_embeddings, prior_samples_var: prior_samples})
                step += 1
                if step % 10 == 0:
                    loss_cur = self.sess.run(loss, feed_dict={real_embedding_var: real_embeddings, prior_samples_var: prior_samples})
                    rel_imp = abs(loss_cur - loss_prev) / abs(loss_prev)
                    if rel_imp < 1e-2:
                        break
                    loss_prev = loss_cur
            loss_final = self.sess.run(loss, feed_dict={real_embedding_var: real_embeddings, prior_samples_var: prior_samples})
            if loss_final < best_of_runs:
                best_of_runs = loss_final
                proj_mat = self.sess.run(tf.concat([v1, v2], 1))
                #dot_prod = tf.reduce_sum(tf.multiply(u, v)).eval()

        real_projections = np.matmul(real_embeddings, proj_mat)
        prior_projections = np.matmul(prior_samples, proj_mat)

        return real_projections, prior_projections



    def sample_data(self, batch_size=100, test=False):
        # returns random sample of batch_size elements from the (by default train, but optionally test) data
        if test is False:
            return self.train_data[np.random.choice(range(len(self.train_data)), batch_size, replace=False)]
        else:
            return self.test_data[np.random.choice(range(len(self.test_data)), batch_size, replace=False)]

    def optimize(self, batch_size=100, iterations=60001, learning_rate=1e-3, plot_all=True, save_model=True):
        for i in range(iterations):
            self.sess.run(
                self.train_step,
                feed_dict={self.learning_rate: learning_rate,
                           self.input: self.sample_data(self.mini_batch_size)}
                )


            if i % 100 == 0:
                print("Iteration %i" % i)
                MMD_IMQ_loss, L1_loss, reconstruction_loss, total_loss = self.sess.run([self.MMD_IMQ, self.L1_loss, self.reconstruction_loss, self.objective],
                                                                  feed_dict={self.input: self.sample_data(batch_size)})
                with open("training_losses.log", "a") as training_loss_log:
                    training_loss_log.write("\nIteration %i \t MMD_IMQ loss: %g \t L1_loss loss: %g \t Reconstruction loss: %g \t Total: %g" %
                                            (i, MMD_IMQ_loss, L1_loss, reconstruction_loss, total_loss))
                print("Train losses: MMD_IMQ loss: %g \t L1_loss loss: %g \t Reconstruction loss: %g \t Total: %g" %
                      (MMD_IMQ_loss, L1_loss, reconstruction_loss, total_loss))
                self.objective_losses.append(total_loss)

                test_MMD_IMQ_loss, test_L1_loss, test_reconstruction_loss, test_total_loss = self.sess.run([self.MMD_IMQ, self.L1_loss, self.reconstruction_loss, self.objective],
                                                                        feed_dict={self.input: self.sample_data(batch_size, test=True)})
                with open("test_losses.log", "a") as test_loss_log:
                    test_loss_log.write("\nIteration %i \t MMD_IMQ loss: %g \t L1_loss loss: %g \t Reconstruction loss: %g \t Total: %g" %
                                        (i, test_MMD_IMQ_loss, L1_loss, test_reconstruction_loss, test_total_loss))

                print("Test losses: MMD_IMQ: %g \t L1_loss: %g \t Reconstruction: %g \t Total: %g" % (test_MMD_IMQ_loss,
                                                                                                      test_L1_loss,
                                                                                   test_reconstruction_loss,
                                                                                   test_total_loss))
                self.test_losses.append(test_total_loss)

                train_logvars = self.sess.run(self.z_logvar,
                                              feed_dict={self.input: self.sample_data(batch_size=1000)}).mean(axis=0)
                test_logvars = self.sess.run(self.z_logvar,
                                             feed_dict={self.input: self.sample_data(batch_size=1000, test=True)}).mean(axis=0)

                with open("average_logvars.log", "a") as average_logvars_log:
                    average_logvars_log.write("\nIteration %i \n Train average logvars:" % i)
                    average_logvars_log.write(np.array2string(train_logvars))
                    average_logvars_log.write("\nTest average logvars:")
                    average_logvars_log.write(np.array2string(test_logvars))


                if len(self.objective_losses) > 20:
                    running_average_diff = np.mean(self.objective_losses[-10:]) - np.mean(self.objective_losses[-20:-10])
                    with open("training_losses.log", "a") as training_loss_log:
                        training_loss_log.write("\t Running average difference: %g" % running_average_diff)

            if i > 0 and i % 60000 == 0:
                if plot_all is True:
                    self.plot_all(str(i))

            if i > 0 and i % 60000 == 0:
                self.disentanglement.do_all(i)

            if i > 0 and i % 60000 == 0:
                if save_model is True:
                    save_path = self.saver.save(self.sess, "saved_model/model", global_step=i)
                    print("Model saved to: %s" % save_path)

                    # If the loss isn't going down anymore, reduce the learning rate
            if i == 15000:
                learning_rate = 7e-4
                print("Reducing learning rate to %g" % learning_rate)
                with open("training_losses.log", "a") as training_loss_log:
                    training_loss_log.write("\nReducing learning rate to %g" % learning_rate)
            if i == 30000:
                learning_rate = 3e-4
                print("Reducing learning rate to %g" % learning_rate)
                with open("training_losses.log", "a") as training_loss_log:
                    training_loss_log.write("\nReducing learning rate to %g" % learning_rate)
            if i == 45000:
                learning_rate = 1e-4
                print("Reducing learning rate to %g" % learning_rate)
                with open("training_losses.log", "a") as training_loss_log:
                    training_loss_log.write("\nReducing learning rate to %g" % learning_rate)


        with open("training_losses.log", "a") as training_loss_log:
            training_loss_log.write("\nTerminating training! Model saved to: %s" % save_path)
        return

    def encode(self, images):
        return self.sess.run(self.z_mean, feed_dict={self.input: images})

    def _encoder_init(self):
        if len(self.input.shape) == 3: # then we have to explicitly add single channel at end
            self.x_reshape = tf.reshape(self.input, shape=(-1,) + self.train_data.shape[1:] + (1,))
        else:
            self.x_reshape = self.input

        self.x_flattened = tf.reshape(self.input, shape=[-1, np.prod(self.data_dims)])

        self.Q_FC1 = tf.layers.dense(inputs=self.x_flattened, units=1200, activation=tf.nn.relu)
        self.Q_FC2 = tf.layers.dense(inputs=self.Q_FC1, units=1200, activation=tf.nn.relu)

        # tanh activation because everything should be in interval [-1, 1]
        self.z_mean = tf.layers.dense(inputs=self.Q_FC2, units=self.K, activation=tf.nn.tanh, name="z_mean")
        self.z_logvar = tf.layers.dense(inputs=self.Q_FC2, units=self.K, name="z_logvar")


        # We want to add random noise in the interval [-t, +t]
        # and for our L1 loss to encourage not using dimentions by setting this to [-1, +1]
        # so make eps ~ [-1, +1] on every dimension so that exp(0) * eps ~[-1, 1]
        self.eps = tf.random_normal(shape=tf.shape(self.z_mean))
        self.z_sample = tf.add(self.z_mean, tf.exp(tf.clip_by_value(self.z_logvar, -16, 10) / 2) * self.eps, name="z_sample")

    def _decoder_init(self):
        self.P_FC1 = tf.layers.dense(inputs=self.z_sample, units=1200, activation=tf.nn.tanh)
        self.P_FC2 = tf.layers.dense(inputs=self.P_FC1, units=1200, activation=tf.nn.tanh)
        self.P_FC3 = tf.layers.dense(inputs=self.P_FC2, units=1200, activation=tf.nn.tanh)
        if self.data_dims[0] == 64:
            self.x_logits = tf.layers.dense(inputs=self.P_FC3, units=4096, name="x_logits")
        elif self.data_dims[0] == 32:
            self.x_logits = tf.layers.dense(inputs=self.P_FC3, units=1024, name="x_logits")

        self.x_logits_img_shape = tf.reshape(self.x_logits, [-1, self.data_dims[0], self.data_dims[1], 1], name="x_logits_img_shape")

    def _prior_init(self):
        self.z_prior_sample = tf.random_normal(shape=tf.shape(self.z_sample), name="z_prior_sample") 

    def _mmd_init(self, C):
        batch_size = tf.shape(self.input)[0]
        batch_size_float = tf.cast(batch_size, tf.float32)


        # ===== Inverse multiquadratic kernel MMD ============

        C_const = tf.cast(tf.constant(C),
                         tf.float32)  # This is also a tunable parameter, 2*K is recommended in the paper

        prior_tensor_ax0_rep = tf.tile(self.z_prior_sample[None, :, :], [batch_size, 1, 1])
        prior_tensor_ax1_rep = tf.tile(self.z_prior_sample[:, None, :], [1, batch_size, 1])
        q_tensor_ax0_rep = tf.tile(self.z_sample[None, :, :], [batch_size, 1, 1])
        q_tensor_ax1_rep = tf.tile(self.z_sample[:, None, :], [1, batch_size, 1])

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

    def _optimizer_init(self):
        MMD_1 = self._mmd_init(0.001*self.K)
        MMD_2 = self._mmd_init(0.005*self.K)
        MMD_3 = self._mmd_init(0.01*self.K)
        MMD_4 = self._mmd_init(0.05*self.K)
        MMD_5 = self._mmd_init(0.1*self.K)
        MMD_6 = self._mmd_init(0.2*self.K)
        MMD_7 = self._mmd_init(0.5*self.K)
        MMD_8 = self._mmd_init(0.8*self.K)
        MMD_9 = self._mmd_init(1.0*self.K)
        MMD_10 = self._mmd_init(2.0*self.K)

        self.MMD_IMQ = MMD_1 + MMD_2 + MMD_3 + MMD_4 + MMD_5 + MMD_6 + MMD_7 + MMD_8 + MMD_9 + MMD_10



        # ===== L1 loss on log-variances ============
        self.L1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.z_logvar), axis=1))

        # ===== Reconstruction Loss ==================

        self.reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.x_logits, labels=self.x_flattened), axis=1))

        # ===== Totla Loss ==================

        self.objective = (self.LAMBDA_IMQ * self.MMD_IMQ) + (self.LAMBDA_L1 * self.L1_loss) + self.reconstruction_loss
        self.learning_rate = tf.placeholder(tf.float32)

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.objective)

    def _plot_all_init(self):
        # Number of random codes that we will fix to use for visualising the network's ability to generate new images
        n_random_codes = 60
        self.random_codes = self.sess.run(self.z_prior_sample,
                                          feed_dict={self.z_sample: np.random.randn(n_random_codes, self.K)}
                                          )
        self.train_reconstruction_images = self.sample_data(batch_size=30)
        self.test_reconstruction_images = self.sample_data(batch_size=30, test=True)

        # Adapted from Ilya's https://github.com/tolstikhin/adagan/blob/master/pot.py add_least_gaussian2d_ops
        with tf.variable_scope("least_gaussian_2d_subspace"):
            real_embedding_var = tf.placeholder(tf.float32, shape=[None, self.K], name="real_embedding_var")
            prior_samples_var = tf.placeholder(tf.float32, shape=[None, self.K], name="prior_samples_var")
            v1 = tf.get_variable("v1", [self.K, 1], tf.float32, tf.random_normal_initializer(stddev=1.0))
            v2 = tf.get_variable("v2", [self.K, 1], tf.float32, tf.random_normal_initializer(stddev=1.0))
            npoints = tf.cast(tf.shape(real_embedding_var)[0], tf.int32)

            # make sure matrix [v1, v2] is orthogonal with unit vectors
            v1_norm = tf.nn.l2_normalize(v1, 0)
            dotprod = tf.reduce_sum(tf.multiply(v2, v1_norm))
            v2_ort = v2 - dotprod * v1_norm
            v2_norm = tf.nn.l2_normalize(v2_ort, 0)
            Mproj = tf.concat([v1_norm, v2_norm], 1)
            real_embeddings_proj = tf.matmul(real_embedding_var, Mproj)  # dimension batch_size x self.K
            prior_samples_proj = tf.matmul(prior_samples_var, Mproj)  # dimension batch_size x self.K
            a = tf.eye(npoints) - tf.ones([npoints, npoints]) / tf.cast(npoints, tf.float32)

            b_real_embeddings = tf.matmul(real_embeddings_proj, tf.matmul(a, a), transpose_a=True)
            b_real_embeddings = tf.matmul(b_real_embeddings, real_embeddings_proj)
            covhat_real_embeddings = b_real_embeddings / (tf.cast(npoints, tf.float32) - 1)

            b_prior_samples_proj = tf.matmul(prior_samples_proj, tf.matmul(a, a), transpose_a=True)
            b_prior_samples_proj = tf.matmul(b_prior_samples_proj, prior_samples_proj)
            covhat_prior_samples_proj = b_prior_samples_proj / (tf.cast(npoints, tf.float32) - 1)

            # l2 distance between real embeddings cov and prior samples cos
            projloss =  tf.reduce_sum(tf.square(covhat_real_embeddings - covhat_prior_samples_proj))
            # Also account for the first moment, i.e. expected value
            projloss += tf.abs( tf.reduce_sum(tf.square(tf.reduce_mean(covhat_prior_samples_proj, 0))) - tf.reduce_sum(tf.square(tf.reduce_mean(covhat_real_embeddings, 0))) )
            # We are maximizing
            projloss = -projloss
            optim = tf.train.AdamOptimizer(0.001, 0.9)
            optim = optim.minimize(projloss, var_list=[v1, v2])

            self._proj_v1 = v1_norm
            self._proj_v2 = v2_norm
            self._proj_real_embedding_var = real_embedding_var
            self._proj_prior_samples_var = prior_samples_var
            self._proj_loss = projloss
            self._proj_optim = optim


def slerp(v1, v2, steps):
    """
    :param v1: start point, 1xN numpy array
    :param v2: end point, 1xN numpy array
    :param steps: number of steps to interpolate along
    :return: Returns the vectors along the spherical interpolation starting at v1 and ending at v2
    Formula taken from wikipedia article https://en.wikipedia.org/wiki/Slerp
    """

    assert v1.shape == v2.shape
    assert v1.shape[0] == 1

    v1_norm = v1 / np.sqrt(np.sum(v1 ** 2))
    v2_norm = v2 / np.sqrt(np.sum(v2 ** 2))
    omega = np.arccos(np.dot(v1_norm, v2_norm.T))
    t = np.linspace(1, 0, steps)

    return (v1 * (np.sin(t * omega) / np.sin(omega)).T) + (v2 * (np.sin((1-t) * omega) / np.sin(omega)).T)

def lerp(v1, v2, steps):
    """
    :param v1: start point, 1xN numpy array
    :param v2: end point, 1xN numpy array
    :param steps: number of steps to interpolate along
    :return: Returns the vectors along the linear interplotaitno starting at v1 and ending at v2
    """

    assert v1.shape == v2.shape
    assert v1.shape[0] == 1

    t = np.linspace(1, 0, steps)[:, None]

    return (v1 * t) + (v2 * (1-t))

if __name__=="__main__":
    DATA_FOLDER = "./"

    DATASET, LATENT_DIMENSIONALITY, LAMBDA_IMQ, LAMBDA_L1, BATCH_SIZE = [str(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])]

    os.chdir(DATASET)
    experiment_file_directory = str(time.time()) + str(np.random.rand())
    os.mkdir(experiment_file_directory)
    os.chdir(experiment_file_directory)
    os.mkdir("output/")
    os.mkdir("output/plots/")
    os.mkdir("saved_model")

    '''
    LATENT_DIMENSIONALITY: dimensionality of latent space
    LAMBDA_IMQ: weighting of the inverse multiquadratic MMD term
    LAMBDA_L1: weighting of the L1 penalty on logvariances (to stop them getting to negative).

    '''


    if DATASET == 'dsprites':
        dsprites_zip = np.load(DATA_FOLDER + "dsprites.npz", encoding='bytes')
        imgs = dsprites_zip['imgs']
        data_in_order = imgs.copy()
        np.random.shuffle(imgs)
        n_data = len(imgs)
        train_data = imgs[0:((9*n_data)//10)]
        test_data = imgs[((9*n_data)//10):]

    # shapes are n_data x 32 x 32

    wae = WAE_MMD(train_data, test_data, data_in_order, LATENT_DIMENSIONALITY, LAMBDA_IMQ, LAMBDA_L1, BATCH_SIZE)


    with open("parameters.txt", "a") as params_file:
        params_file.write("Dataset used: %s\n" % DATASET)
        params_file.write("Parameters: \t LATENT_DIMENSIONALITY %i \t LAMBDA_IMQ %g \t LAMBDA_L1 %g \t BATCH_SIZE %i" %
                          (LATENT_DIMENSIONALITY, LAMBDA_IMQ, LAMBDA_L1, BATCH_SIZE))

    with open("README.txt", "a") as readme_file:
        readme_file.write(readme)

    wae.optimize()

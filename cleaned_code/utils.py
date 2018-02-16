fixed_import tensorflow as tf
import numpy as np

def print_log_information(model, it):
    # print train and test error to logs
    train_batch = model.sample_minibatch()
    test_batch = model.sample_minibatch(test=True)
    batches = [train_batch, test_batch, model.fixed_test_sample]
    files = ['loss_train.log', 'loss_test_random.log', 'loss_test_fixed.log']
    losses = [model.losses_train, model.losses_test_random, model.losses_test_fixed]

    assert len(batches) == len(files)

    for j in range(len(batches)):
        loss_reconstruction, loss_regulariser, total_loss = model.sess.run([model.loss_reconstruction,
                                                                           model.loss_regulariser,
                                                                           model.total_loss],
                                                                           feed_dict={model.input: batches[j]})
        losses[j].append(total_loss)

        if hasattr(model, 'z_logvar_loss'):
            z_logvar_loss = model.sess.run(model.z_logvar_loss, feed_dict={model.input: batches[j]})
            with open(files[j], "a") as training_loss_log:
                training_loss_log.write("\nIteration %i \t Regulariser loss: %g \t Logvar penalty loss: %g \t Reconstruction loss: %g \t Total: %g" %
                                        (it, loss_regulariser, z_logvar_loss, loss_reconstruction, total_loss))
        else:
            with open(files[j], "a") as training_loss_log:
                training_loss_log.write("\nIteration %i \t Logvar penalty loss: %g \t Reconstruction loss: %g \t Total: %g" %
                                        (it, loss_regulariser, loss_reconstruction, total_loss))

    # TODO print max, min and average logvars for each dimension

def plot_all_init(model):
    # Adapted from Ilya's https://github.com/tolstikhin/adagan/blob/master/pot.py add_least_gaussian2d_ops
    with tf.variable_scope("least_gaussian_2d_subspace"):
        real_embedding_var = tf.placeholder(tf.float32, shape=[None, model.z_dim], name="real_embedding_var")
        prior_samples_var = tf.placeholder(tf.float32, shape=[None, model.z_dim], name="prior_samples_var")
        v1 = tf.get_variable("v1", [model.z_dim, 1], tf.float32, tf.random_normal_initializer(stddev=1.0))
        v2 = tf.get_variable("v2", [model.z_dim, 1], tf.float32, tf.random_normal_initializer(stddev=1.0))
        npoints = tf.cast(tf.shape(real_embedding_var)[0], tf.int32)

        # make sure matrix [v1, v2] is orthogonal with unit vectors
        v1_norm = tf.nn.l2_normalize(v1, 0)
        dotprod = tf.reduce_sum(tf.multiply(v2, v1_norm))
        v2_ort = v2 - dotprod * v1_norm
        v2_norm = tf.nn.l2_normalize(v2_ort, 0)
        Mproj = tf.concat([v1_norm, v2_norm], 1)
        real_embeddings_proj = tf.matmul(real_embedding_var, Mproj)  # dimension batch_size x z_dim
        prior_samples_proj = tf.matmul(prior_samples_var, Mproj)  # dimension batch_size x z_dim
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

        model._proj_v1 = v1_norm
        model._proj_v2 = v2_norm
        model._proj_real_embedding_var = real_embedding_var
        model._proj_prior_samples_var = prior_samples_var
        model._proj_loss = projloss
        model._proj_optim = optim

def plot_all(model, filename_appendage):
    fixed_test_sample = model.fixed_test_sample[0:30]
    fixed_train_sample = model.fixed_train_sample[0:30]
    fixed_codes = model.fixed_codes[0:30]

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

    train_reconstruction_images = model.decode(model.encode(fixed_train_sample))

    for j in range(0, 60, 2):
        ax_real = plt.Subplot(fig, inner[j])
        ax_real.imshow(fixed_train_sample[j//2, :, :], cmap="gray")
        ax_real.axis("off")
        fig.add_subplot(ax_real)

        ax_recon = plt.Subplot(fig, inner[j + 1])
        ax_recon.imshow(train_reconstruction_images[j//2, :, :], cmap="gray")
        ax_recon.axis("off")
        fig.add_subplot(ax_recon)

    sample_train_images = model.sample_minibatch(batch_size=20)
    sample_reconstructed_images = model.decode(model.encode(sample_train_images))

    for j in range(60, 100, 2):
        ax_real = plt.Subplot(fig, inner[j])
        ax_real.imshow(sample_train_images[j//2 - 30, :, :], cmap="gray")
        ax_real.axis("off")
        fig.add_subplot(ax_real)

        ax_recon = plt.Subplot(fig, inner[j + 1])
        ax_recon.imshow(sample_reconstructed_images[j//2 - 30, :, :], cmap="gray")
        ax_recon.axis("off")
        fig.add_subplot(ax_recon)






    # Plot reconstructions of test images
    inner = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
    test_reconstruction_images = model.decode(model.encode(fixed_test_sample))

    for j in range(0, 60, 2):
        ax_real = plt.Subplot(fig, inner[j])
        ax_real.imshow(fixed_test_sample[j//2, :, :], cmap="gray")
        ax_real.axis("off")
        fig.add_subplot(ax_real)

        ax_recon = plt.Subplot(fig, inner[j + 1])
        ax_recon.imshow(test_reconstruction_images[j//2, :, :], cmap="gray")
        ax_recon.axis("off")
        fig.add_subplot(ax_recon)

    sample_test_images = model.sample_minibatch(batch_size=20, test=True)
    sample_reconstructed_images = model.decode(model.encode(sample_test_images))

    for j in range(60, 100, 2):
        ax_real = plt.Subplot(fig, inner[j])
        ax_real.imshow(sample_test_images[j//2 - 30, :, :], cmap="gray")
        ax_real.axis("off")
        fig.add_subplot(ax_real)

        ax_recon = plt.Subplot(fig, inner[j + 1])
        ax_recon.imshow(sample_reconstructed_images[j//2 - 30, :, :], cmap="gray")
        ax_recon.axis("off")
        fig.add_subplot(ax_recon)



    # Plot images generated from fixed random samples
    inner = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=outer[2], wspace=0.1, hspace=0.1)

    fixed_random_sample_images = model.decode(fixed_codes)

    for j in range(0, 60, 1):
        ax = plt.Subplot(fig, inner[j])
        ax.imshow(fixed_random_sample_images[j, :, :], cmap="gray")
        ax.axis("off")
        fig.add_subplot(ax)


    # model.z_sample
    random_codes = model.sample_codes(batch_size=40)
    varying_random_sample_images = model.decode(random_codes)

    for j in range(60, 100, 1):
        ax = plt.Subplot(fig, inner[j])
        ax.imshow(varying_random_sample_images[j - 60, :, :], cmap="gray")
        ax.axis("off")
        fig.add_subplot(ax)



    # ========== Interpolations ===========

    # Training data
    inner = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=outer[3], wspace=0.1, hspace=0.1)

    for i in range(5):
        m1 = model.encode(fixed_train_sample[i][None,:,:,:], mean=True)
        m2 = model.encode(fixed_train_sample[i+1][None,:,:,:], mean=True)
        embeddings = lerp(m1, m2, 10) # LINEAR for uniform box
        outputs = model.decode(embeddings)

        for j in range(10):
            ax = plt.Subplot(fig, inner[10*i + j])
            ax.imshow(outputs[j, :, :], cmap='gray')
            ax.axis("off")
            fig.add_subplot(ax)

    for i in range(5, 10):
        m1 = model.encode(model.sample_data(batch_size=1), mean=True)
        m2 = model.encode(model.sample_data(batch_size=1), mean=True)
        embeddings = lerp(m1, m2, 10)
        outputs = model.decode(embeddings)

        for j in range(10):
            ax = plt.Subplot(fig, inner[10*i + j])
            ax.imshow(outputs[j, :, :], cmap='gray')
            ax.axis("off")
            fig.add_subplot(ax)


    # Test data
    inner = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=outer[4], wspace=0.1, hspace=0.1)

    for i in range(5):
        m1 = model.encode(fixed_test_sample[i][None,:,:,:], mean=True)
        m2 = model.encode(fixed_test_sample[i+1][None,:,:,:], mean=True)
        embeddings = lerp(m1, m2, 10) # LINEAR for uniform box
        outputs = model.decode(embeddings)

        for j in range(10):
            ax = plt.Subplot(fig, inner[10*i + j])
            ax.imshow(outputs[j, :, :], cmap='gray')
            ax.axis("off")
            fig.add_subplot(ax)

    for i in range(5, 10):
        m1 = model.encode(model.sample_data(batch_size=1, test=True), mean=True)
        m2 = model.encode(model.sample_data(batch_size=1, test=True), mean=True)
        embeddings = lerp(m1, m2, 10)
        outputs = model.decode(embeddings)

        for j in range(10):
            ax = plt.Subplot(fig, inner[10*i + j])
            ax.imshow(outputs[j, :, :], cmap='gray')
            ax.axis("off")
            fig.add_subplot(ax)




    # Random samples from the latent space
    inner = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=outer[5], wspace=0.1, hspace=0.1)

    for i in range(5):
        m1 = fixed_codes[i][None,:]
        m2 = fixed_codes[i+1][None,:]
        embeddings = lerp(m1, m2, 10)
        outputs = model.decode(embeddings)

        for j in range(10):
            ax = plt.Subplot(fig, inner[10*i + j])
            ax.imshow(outputs[j, :, :], cmap='gray')
            ax.axis("off")
            fig.add_subplot(ax)

    for i in range(5, 10):
        m1 = model.sample_codes(batch_size=1)
        m2 = model.sample_codes(batch_size=1)
        embeddings = lerp(m1, m2, 10)
        outputs = model.decode(embeddings)

        for j in range(10):
            ax = plt.Subplot(fig, inner[10*i + j])
            ax.imshow(outputs[j, :, :], cmap='gray')
            ax.axis("off")
            fig.add_subplot(ax)


    # ========== Walks along each axis direction ===========
    if model.opts['plot_axis_walks'] is True:
        axis_walk_range = model.opts['axis_walk_range']
        # Training data
        inner = gridspec.GridSpecFromSubplotSpec(model.K, 10, subplot_spec=outer[6], wspace=0.1, hspace=0.1)

        mean = model.encode(model.sample_minibatch(batch_size=1), mean=True)

        for axis in range(model.z_dim):
            repeat_mean = np.repeat(mean, 10, axis=0)
            repeat_mean[:, axis] = np.linspace(-axis_walk_range,axis_walk_range,10)
            outputs = model.decode(repeat_mean)

            for j in range(10):
                ax = plt.Subplot(fig, inner[10*axis + j])
                ax.imshow(outputs[j, :, :], cmap='gray')
                ax.axis("off")
                fig.add_subplot(ax)


        # Test data
        inner = gridspec.GridSpecFromSubplotSpec(model.z_dim, 10, subplot_spec=outer[7], wspace=0.1, hspace=0.1)

        mean = model.encode(model.sample_minibatch(batch_size=1, test=True), mean=True)
        for axis in range(model.z_dim):
            repeat_mean = np.repeat(mean, 10, axis=0)
            repeat_mean[:, axis] = np.linspace(-axis_walk_range,axis_walk_range,10)
            outputs = model.decode(repeat_mean)

            for j in range(10):
                ax = plt.Subplot(fig, inner[10*axis + j])
                ax.imshow(outputs[j, :, :], cmap='gray')
                ax.axis("off")
                fig.add_subplot(ax)



        # Random samples from the latent space
        inner = gridspec.GridSpecFromSubplotSpec(model.z_dim, 10, subplot_spec=outer[8], wspace=0.1, hspace=0.1)

        mean = mode.sample_codes(batch_size=1)
        for axis in range(model.K):
            repeat_mean = np.repeat(mean, 10, axis=0)
            repeat_mean[:, axis] = np.linspace(-axis_walk_range,axis_walk_range,10)
            outputs = model.sess.run(tf.nn.sigmoid(model.x_logits_img_shape),
                                    feed_dict={model.z_sample: repeat_mean})
            for j in range(10):
                ax = plt.Subplot(fig, inner[10*axis + j])
                ax.imshow(outputs[j, :, :], cmap='gray')
                ax.axis("off")
                fig.add_subplot(ax)



    # ========== Training and test error plots ===========
    if model.opts['plot_losses'] is True:
        inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[9], wspace=0.1, hspace=0.1)

        ax = plt.Subplot(fig, inner[0])
        if len(model.losses_train) > 5:
            ax.plot(np.log(model.losses_train[3:]), linewidth=7.0)
            ax.plot(np.log(model.losses_test_fixed[3:]), linewidth=7.0)
        else:
            ax.plot(np.log(model.losses_train), linewidth=7.0)
            ax.plot(np.log(model.losses_test_fixed), linewidth=7.0)
        ax.legend(["Log training loss", "Log test loss"], prop={'size': 40})
        fig.add_subplot(ax)


        # ========== Most non-gaussian 2d subspace ===========

        inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[10], wspace=0.1, hspace=0.1)

        real_embeddings = model.encode(model.sample_minibatch(batch_size=100))
        prior_samples = model.sample_codes(batch_size=100)

        real_projections, prior_projections = model.least_gaussian_2d_subspace(real_embeddings, prior_samples)

        ax = plt.Subplot(fig, inner[0])
        ax.scatter(real_projections[:,0], real_projections[:,1], s=50)
        ax.scatter(prior_projections[:,0], prior_projections[:,1], s=50)
        ax.legend(["Q(Z)", "P(Z)"], prop={'size': 40})
        fig.add_subplot(ax)





    fig.savefig("output/plots/" + filename_appendage + ".png")
    plt.close(fig)
    plt.close("all")

    return

def _least_gaussian_2d_subspace(model, real_embeddings, prior_samples):
    # adapted from ilya's https://github.com/tolstikhin/adagan/blob/master/pot.py#L1349
    real_embedding_var = model._proj_real_embedding_var
    prior_samples_var = model._proj_prior_samples_var
    optim = model._proj_optim
    loss = model._proj_loss
    v1 = model._proj_v1
    v2 = model._proj_v2
    #proj_mat = tf.concat([v, u], 1).eval()
    #dot_prod = -1
    best_of_runs = 10e5 # Any positive value would do
    updated = False

    for _start in range(3):
        # We will run 3 times from random inits
        loss_prev = 10e5 # Any positive value would do
        proj_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="least_gaussian_2d_subspace")
        model.sess.run(tf.variables_initializer(proj_vars))
        step = 0
        for _ in range(5000):
            model.sess.run(optim, feed_dict={real_embedding_var:real_embeddings, prior_samples_var: prior_samples})
            step += 1
            if step % 10 == 0:
                loss_cur = model.sess.run(loss, feed_dict={real_embedding_var: real_embeddings, prior_samples_var: prior_samples})
                rel_imp = abs(loss_cur - loss_prev) / abs(loss_prev)
                if rel_imp < 1e-2:
                    break
                loss_prev = loss_cur
        loss_final = model.sess.run(loss, feed_dict={real_embedding_var: real_embeddings, prior_samples_var: prior_samples})
        if loss_final < best_of_runs:
            best_of_runs = loss_final
            proj_mat = model.sess.run(tf.concat([v1, v2], 1))
            #dot_prod = tf.reduce_sum(tf.multiply(u, v)).eval()

    real_projections = np.matmul(real_embeddings, proj_mat)
    prior_projections = np.matmul(prior_samples, proj_mat)

    return real_projections, prior_projections

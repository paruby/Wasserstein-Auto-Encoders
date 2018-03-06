import tensorflow as tf
import numpy as np
import os
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import zipfile
import pickle

ROOT_FOLDER = os.getcwd()

def save_opts(model):
    file_path = ROOT_FOLDER + "/" + model.opts['experiment_path'] + "/opts.txt"
    with open(file_path, "a") as opts_file:
        opts_file.write("{\n" + "\n".join("{}: {}".format(k, v) for k, v in model.opts.items()) + "\n}")
    pickle_path = ROOT_FOLDER + "/" + model.opts['experiment_path'] + "/opts.pickle"
    with open(pickle_path, "wb") as opts_file:
        pickle.dump(model.opts, opts_file, protocol=pickle.HIGHEST_PROTOCOL)


def copy_all_code(model):
    zip_path = ROOT_FOLDER + "/" + model.opts['experiment_path'] + "/code.zip"
    zipf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    files = [f for f in os.listdir('.') if os.path.isfile(f) and f[-3:] == '.py']
    for f in files:
        zipf.write(f)
    zipf.close()


def load_disentanglement_data_dsprites():
    dataset_zip = np.load(ROOT_FOLDER + "/datasets/dsprites.npz", encoding='bytes')
    imgs = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']
    latents_classes = dataset_zip['latents_classes']
    metadata = dataset_zip['metadata'][()]
    latents_sizes = metadata[b'latents_sizes']  # for some reason, I was getting a key error here so hard coded in the next line.
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                    np.array([1,])))
    return imgs, latents_sizes, latents_bases

def load_data(model, seed=None):
    # set seed
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)

    dataset = model.opts['dataset']
    if dataset == 'dsprites':
        try:
            dsprites_zip = np.load(ROOT_FOLDER + '/datasets/dsprites.npz', encoding='bytes')
        except FileNotFoundError:
            print("Dataset file does not exist. You can download it here: https://github.com/deepmind/dsprites-dataset/ Save dsprites_ndarray_....npz in datasets folder as dsprites.npz")
        data = dsprites_zip['imgs']

    elif dataset == 'cifar':
        try:
            data = np.load(ROOT_FOLDER + '/datasets/cifar10.npy')
        except FileNotFoundError:
            print("Dataset file does not exist.")

    elif dataset == 'celebA':
        try:
            data = np.load(ROOT_FOLDER + '/datasets/celebA.npy')
        except FileNotFoundError:
            print("Dataset file does not exist. You can download it here: https://www.dropbox.com/sh/flu98x7xghw2i59/AAC-eDY7TS9V54AxCtvBjGTAa?dl=0 Save celebA.npy in datasets folder")

    elif dataset == 'celebA_mini':
        try:
            data = np.load(ROOT_FOLDER + '/datasets/celebA_mini.npy')
        except FileNotFoundError:
            print("Dataset file does not exist. You can download it here: https://www.dropbox.com/sh/flu98x7xghw2i59/AAC-eDY7TS9V54AxCtvBjGTAa?dl=0 Save celebA_mini.npy in datasets folder")

    elif dataset == 'fading_squares':
        try:
            data = np.load(ROOT_FOLDER + '/datasets/fading_squares.npy')
        except FileNotFoundError:
            print("Dataset file does not exist. You can download it here: https://www.dropbox.com/sh/flu98x7xghw2i59/AAC-eDY7TS9V54AxCtvBjGTAa?dl=0 Save fading_squares.npy in datasets folder")

    elif dataset == 'grassli':
        try:
            data = np.load(ROOT_FOLDER + '/datasets/grassli.npy')
            data = data / 255 # pixels should be in [0,1], not [0,255]
        except FileNotFoundError:
            print("Dataset file does not exist. You can get this from Ilya!")
    # last channel should be 1d if images are black/white
    if len(data.shape) == 3:
        data = data[:, :, :, None]
    n_data = len(data)
    np.random.shuffle(data)
    # 90% test/train split
    train_data = data[0:((9*n_data)//10)]
    test_data = data[((9*n_data)//10):]

    # reset seed
    if seed is not None:
        np.random.set_state(st0)
    return train_data, test_data

def create_directories(model):
    pathlib.Path(model.experiment_path).mkdir(parents=True, exist_ok=True)
    os.chdir(model.experiment_path)
    pathlib.Path('output/plots/').mkdir(parents=True, exist_ok=True)
    pathlib.Path('checkpoints').mkdir(exist_ok=True)

def print_log_information(model, it):
    # print train and test error to logs
    train_batch = model.sample_minibatch()
    test_batch = model.sample_minibatch(test=True)
    batches = [train_batch, test_batch, model.fixed_test_sample]
    files = ['loss_train.log', 'loss_test_random.log', 'loss_test_fixed.log']
    losses = [model.losses_train, model.losses_test_random, model.losses_test_fixed]

    assert len(batches) == len(files)

    for j in range(len(batches)):
        loss_reconstruction, loss_regulariser, loss_total = model.sess.run([model.loss_reconstruction,
                                                                           model.loss_regulariser,
                                                                           model.loss_total],
                                                                           feed_dict={model.input: batches[j]})
        losses[j].append(loss_total)

        if hasattr(model, 'z_logvar_loss'):
            z_logvar_loss = model.sess.run(model.z_logvar_loss, feed_dict={model.input: batches[j]})
            with open(files[j], "a") as training_loss_log:
                training_loss_log.write("\nIteration %i \t Regulariser loss: %g \t Logvar penalty loss: %g \t Reconstruction loss: %g \t Total: %g" %
                                        (it, loss_regulariser, z_logvar_loss, loss_reconstruction, loss_total))
        else:
            with open(files[j], "a") as training_loss_log:
                training_loss_log.write("\nIteration %i \t Regulariser loss: %g \t Reconstruction loss: %g \t Total: %g" %
                                        (it, loss_regulariser, loss_reconstruction, loss_total))

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
    fixed_codes = model.fixed_codes[0:60]

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
        ax_real.imshow(_imshow_process(fixed_train_sample[j//2]) , cmap="gray", vmin=0, vmax=1)
        ax_real.axis("off")
        fig.add_subplot(ax_real)

        ax_recon = plt.Subplot(fig, inner[j + 1])
        ax_recon.imshow(_imshow_process(train_reconstruction_images[j//2]) , cmap="gray", vmin=0, vmax=1)
        ax_recon.axis("off")
        fig.add_subplot(ax_recon)

    sample_train_images = model.sample_minibatch(batch_size=20)
    sample_reconstructed_images = model.decode(model.encode(sample_train_images))

    for j in range(60, 100, 2):
        ax_real = plt.Subplot(fig, inner[j])
        ax_real.imshow(_imshow_process(sample_train_images[j//2 - 30]) , cmap="gray", vmin=0, vmax=1)
        ax_real.axis("off")
        fig.add_subplot(ax_real)

        ax_recon = plt.Subplot(fig, inner[j + 1])
        ax_recon.imshow(_imshow_process(sample_reconstructed_images[j//2 - 30]) , cmap="gray", vmin=0, vmax=1)
        ax_recon.axis("off")
        fig.add_subplot(ax_recon)






    # Plot reconstructions of test images
    inner = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
    test_reconstruction_images = model.decode(model.encode(fixed_test_sample))

    for j in range(0, 60, 2):
        ax_real = plt.Subplot(fig, inner[j])
        ax_real.imshow(_imshow_process(fixed_test_sample[j//2]) , cmap="gray", vmin=0, vmax=1)
        ax_real.axis("off")
        fig.add_subplot(ax_real)

        ax_recon = plt.Subplot(fig, inner[j + 1])
        ax_recon.imshow(_imshow_process(test_reconstruction_images[j//2]) , cmap="gray", vmin=0, vmax=1)
        ax_recon.axis("off")
        fig.add_subplot(ax_recon)

    sample_test_images = model.sample_minibatch(batch_size=20, test=True)
    sample_reconstructed_images = model.decode(model.encode(sample_test_images))

    for j in range(60, 100, 2):
        ax_real = plt.Subplot(fig, inner[j])
        ax_real.imshow(_imshow_process(sample_test_images[j//2 - 30]) , cmap="gray", vmin=0, vmax=1)
        ax_real.axis("off")
        fig.add_subplot(ax_real)

        ax_recon = plt.Subplot(fig, inner[j + 1])
        ax_recon.imshow(_imshow_process(sample_reconstructed_images[j//2 - 30]) , cmap="gray", vmin=0, vmax=1)
        ax_recon.axis("off")
        fig.add_subplot(ax_recon)



    # Plot images generated from fixed random samples
    inner = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=outer[2], wspace=0.1, hspace=0.1)

    fixed_random_sample_images = model.decode(fixed_codes)

    for j in range(0, 60, 1):
        ax = plt.Subplot(fig, inner[j])
        ax.imshow(_imshow_process(fixed_random_sample_images[j]) , cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        fig.add_subplot(ax)


    # model.z_sample
    random_codes = model.sample_codes(batch_size=40)
    varying_random_sample_images = model.decode(random_codes)

    for j in range(60, 100, 1):
        ax = plt.Subplot(fig, inner[j])
        ax.imshow(_imshow_process(varying_random_sample_images[j - 60]) , cmap="gray", vmin=0, vmax=1)
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
            ax.imshow(_imshow_process(outputs[j, :, :]) , cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            fig.add_subplot(ax)

    for i in range(5, 10):
        m1 = model.encode(model.sample_minibatch(batch_size=1), mean=True)
        m2 = model.encode(model.sample_minibatch(batch_size=1), mean=True)
        embeddings = lerp(m1, m2, 10)
        outputs = model.decode(embeddings)

        for j in range(10):
            ax = plt.Subplot(fig, inner[10*i + j])
            ax.imshow(_imshow_process(outputs[j]) , cmap="gray", vmin=0, vmax=1)
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
            ax.imshow(_imshow_process(outputs[j]) , cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            fig.add_subplot(ax)

    for i in range(5, 10):
        m1 = model.encode(model.sample_minibatch(batch_size=1, test=True), mean=True)
        m2 = model.encode(model.sample_minibatch(batch_size=1, test=True), mean=True)
        embeddings = lerp(m1, m2, 10)
        outputs = model.decode(embeddings)

        for j in range(10):
            ax = plt.Subplot(fig, inner[10*i + j])
            ax.imshow(_imshow_process(outputs[j]) , cmap="gray", vmin=0, vmax=1)
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
            ax.imshow(_imshow_process(outputs[j]) , cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            fig.add_subplot(ax)

    for i in range(5, 10):
        m1 = model.sample_codes(batch_size=1)
        m2 = model.sample_codes(batch_size=1)
        embeddings = lerp(m1, m2, 10)
        outputs = model.decode(embeddings)

        for j in range(10):
            ax = plt.Subplot(fig, inner[10*i + j])
            ax.imshow(_imshow_process(outputs[j]) , cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            fig.add_subplot(ax)


    # ========== Walks along each axis direction ===========
    if model.opts['plot_axis_walks'] is True:
        axis_walk_range = model.opts['axis_walk_range']
        # Training data
        inner = gridspec.GridSpecFromSubplotSpec(model.z_dim, 10, subplot_spec=outer[6], wspace=0.1, hspace=0.1)

        mean = model.encode(model.sample_minibatch(batch_size=1), mean=True)

        for axis in range(model.z_dim):
            repeat_mean = np.repeat(mean, 10, axis=0)
            repeat_mean[:, axis] = np.linspace(-axis_walk_range,axis_walk_range,10)
            outputs = model.decode(repeat_mean)

            for j in range(10):
                ax = plt.Subplot(fig, inner[10*axis + j])
                ax.imshow(_imshow_process(outputs[j]) , cmap="gray", vmin=0, vmax=1)
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
                ax.imshow(_imshow_process(outputs[j]) , cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                fig.add_subplot(ax)



        # Random samples from the latent space
        inner = gridspec.GridSpecFromSubplotSpec(model.z_dim, 10, subplot_spec=outer[8], wspace=0.1, hspace=0.1)

        mean = model.sample_codes(batch_size=1)
        for axis in range(model.z_dim):
            repeat_mean = np.repeat(mean, 10, axis=0)
            repeat_mean[:, axis] = np.linspace(-axis_walk_range,axis_walk_range,10)
            outputs = model.sess.run(tf.nn.sigmoid(model.x_logits_img_shape),
                                    feed_dict={model.z_sample: repeat_mean})
            for j in range(10):
                ax = plt.Subplot(fig, inner[10*axis + j])
                ax.imshow(_imshow_process(outputs[j]) , cmap="gray", vmin=0, vmax=1)
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

        real_embeddings = model.encode(model.sample_minibatch(batch_size=100), mean=False)
        prior_samples = model.sample_codes(batch_size=100)

        real_projections, prior_projections = least_gaussian_2d_subspace(model, real_embeddings, prior_samples)

        ax = plt.Subplot(fig, inner[0])
        ax.scatter(real_projections[:,0], real_projections[:,1], s=50)
        ax.scatter(prior_projections[:,0], prior_projections[:,1], s=50)
        ax.legend(["Q(Z)", "P(Z)"], prop={'size': 40})
        fig.add_subplot(ax)





    fig.savefig("output/plots/" + str(filename_appendage) + ".png")
    plt.close(fig)
    plt.close("all")

    return

def least_gaussian_2d_subspace(model, real_embeddings, prior_samples):
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

def _imshow_process(tensor):
    '''
    imshow throws an error if the color channel has dimension 1.
    This function does nothing if the color channel dimension is NOT 1
    if the color channel dimension is 1, it returns the slice removing this dim
    '''
    assert len(tensor.shape) == 3
    if tensor.shape[-1] == 1:
        return tensor[:,:,0]
    else:
        return tensor


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

def opts_check(model):
    opts = model.opts
    assert type(opts['save_every']) is int
    assert opts['dataset'] in ['fading_squares', 'dsprites', 'celebA', 'celebA_mini', 'grassli', 'cifar']
    if opts['dataset'] != 'dsprites':
        assert 'disentanglement_metric' not in opts
    if 'disentanglement_metric' in opts:
        assert type(opts['disentanglement_metric']) is bool
    assert type(opts['experiment_path']) is str
    assert type(opts['z_dim']) is int
    assert opts['print_log_information'] in [True, False]
    assert (type(opts['make_pictures_every']) is int) or (opts['make_pictures_every'] is None)
    assert type(opts['plot_axis_walks']) is bool
    if opts['plot_axis_walks'] is True: #if z_dim >> 10, plotting axis walks will be long
        assert opts['axis_walk_range'] > 0
    assert type(opts['plot_losses']) is bool
    if opts['plot_losses'] is True:
        assert opts['print_log_information'] is True
    assert type(opts['batch_size']) is int
    assert opts["encoder_architecture"] in ['small_convolutional_celebA', 'FC_dsprites', 'dcgan']
    if opts["encoder_architecture"] == 'dcgan':
        assert type(model.opts['encoder_num_filters']) is int
        assert type(model.opts['encoder_num_layers']) is int
        assert type(model.opts['conv_filter_dim']) is int
    assert opts["decoder_architecture"] in ['small_convolutional_celebA', 'FC_dsprites', 'dcgan', 'dcgan_mod']
    if opts["decoder_architecture"] in ['dcgan', 'dcgan_mod'] :
        assert type(model.opts['decoder_num_filters']) is int
        assert type(model.opts['decoder_num_layers']) is int
        assert type(model.opts['conv_filter_dim']) is int
    assert opts['z_mean_activation'] in ['tanh', None]
    assert opts['encoder_distribution'] in ['deterministic', 'gaussian', 'uniform']
    assert opts['logvar-clipping'] is None or (len(opts['logvar-clipping']) == 2 and all([type(i) is int for i in opts['logvar-clipping']]))
    assert opts['z_prior'] in ['gaussian', 'uniform']
    assert opts['loss_reconstruction'] in ['bernoulli', 'L2_squared', 'L2_squared+adversarial', 'L2_squared+adversarial+l2_filter', 'L2_squared+multilayer_conv_adv', 'L2_squared+adversarial+l2_norm', 'patch_moments']
    if opts['loss_reconstruction'] in ['L2_squared+adversarial', 'L2_squared+adversarial+l2_filter', 'L2_squared+multilayer_conv_adv']:
        assert type(opts['adversarial_cost_n_filters']) is int
        assert type(opts['adversarial_cost_kernel_size']) is int
        assert type(opts['adv_cost_learning_rate_schedule']) is list
        assert all([type(l) is tuple and len(l)==2 for l in opts['adv_cost_learning_rate_schedule']])
        assert all([opts['adv_cost_learning_rate_schedule'][i][1] < opts['adv_cost_learning_rate_schedule'][i+1][1] for i in range(len(opts['adv_cost_learning_rate_schedule'])-1)])
    assert opts['loss_regulariser'] in [None, 'VAE', 'beta_VAE', 'WAE_MMD'] # either KL divergence of VAE or divergence of WAE
    if opts['loss_regulariser'] == 'beta_VAE':
        assert type(opts['beta']) is float
    if opts['loss_regulariser'] == 'WAE_MMD':
        assert type(opts['lambda_imq']) is float
        assert type(opts['IMQ_length_params']) is list # parameters should be scaled according to z_dim
        assert all(type(i) is float for i in opts['IMQ_length_params'])
    assert opts['z_logvar_regularisation'] in [None, "L1", "L2_squared"]
    if opts['z_logvar_regularisation'] is not None:
        assert type(opts['lambda_logvar_regularisation']) is float
    assert opts['optimizer'] in ['adam']
    if opts['optimizer'] == 'adam':
        assert type(opts['learning_rate_schedule']) is list
        assert all([type(l) is tuple and len(l)==2 for l in opts['learning_rate_schedule']])
        assert all([opts['learning_rate_schedule'][i][1] < opts['learning_rate_schedule'][i+1][1] for i in range(len(opts['learning_rate_schedule'])-1)])
        # opts['learning_rate_schedule'] = [(learning_rate, iteration # that this is valid for)]
        # e.g. opts['learning_rate_schedule'] = [(1e-4, 20000), (3e-5, 40000), (1e-5, 60000)]

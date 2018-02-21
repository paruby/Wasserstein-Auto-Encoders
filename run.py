import wae
import argparse
import config

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    help='Dataset to train on: dsprites/celebA/celebA_mini/fading_squares')
parser.add_argument("--z_dim", help='latent space dimensionality', type=int)
parser.add_argument("--lambda_imq", help='Lambda for WAE penalty', type=float)
parser.add_argument("--experiment_path",
                    help="Relative path to where this experiment should save results")
parser.add_argument("--encoder_distribution",
                    help="Encoder distribution: deterministic/gaussian/uniform")
parser.add_argument("--z_prior",
                    help="Prior distribution over latent space: gaussian/uniform")
parser.add_argument("--loss_reconstruction",
                    help="Image reconstruction loss: bernoulli/L2_squared")
parser.add_argument("--loss_regulariser",
                    help="Model type: VAE/beta_VAE/WAE_MMD")
parser.add_argument("--beta", type=float,
                    help="beta parameter for beta_VAE")
parser.add_argument("--disentanglement_metric", type=bool,
                    help="Calculate disentanglement metric")
parser.add_argument("--make_pictures_every", type=int,
                    help="How often to plot random samples and reconstructions")
parser.add_argument("--save_every", type=int,
                    help="How often to save the model")
parser.add_argument("--batch_size", type=int,
                    help="Batch size. Default 100")
parser.add_argument("--encoder_architecture",
                    help="Architecture of encoder: FC_dsprites/small_convolutional_celebA")
parser.add_argument("--decoder_architecture",
                    help="Architecture of decoder: FC_dsprites/small_convolutional_celebA")
parser.add_argument("--z_logvar_regularisation",
                    help="Regularisation on log-variances: None/L1/L2_squared")
parser.add_argument("--lambda_logvar_regularisation", type=float,
                    help="Coefficient of logvariance regularisation")
parser.add_argument("--plot_losses", type=bool,
                    help="Plot losses and least-gaussian-subspace: True/False:")

FLAGS = parser.parse_args()

if __name__ == "__main__":
    if FLAGS.dataset == 'dsprites':
        opts = config.dsprites_opts
    elif FLAGS.dataset == 'fading_squares':
        opts = config.fading_squares_opts
    elif FLAGS.dataset == 'celebA':
        opts = config.celebA_opts
    elif FLAGS.dataset == 'celebA_mini':
        opts = config.celebA_mini_opts
    else:
        assert False, "Invalid dataset"

    if FLAGS.z_dim:
        opts['z_dim'] = FLAGS.z_dim
    if FLAGS.lambda_imq:
        opts['lambda_imq'] = FLAGS.lambda_imq
    if FLAGS.experiment_path:
        opts['experiment_path'] = FLAGS.experiment_path
    if FLAGS.encoder_distribution:
        opts['encoder_distribution'] = FLAGS.encoder_distribution
    if FLAGS.z_prior:
        opts['z_prior'] = FLAGS.z_prior
    if FLAGS.loss_reconstruction:
        opts['loss_reconstruction'] = FLAGS.loss_reconstruction
    if FLAGS.disentanglement_metric:
        opts['disentanglement_metric'] = FLAGS.disentanglement_metric
    if FLAGS.make_pictures_every:
        opts['make_pictures_every'] = FLAGS.make_pictures_every
    if FLAGS.save_every:
        opts['save_every'] = FLAGS.save_every
    if FLAGS.batch_size:
        opts['batch_size'] = FLAGS.batch_size
    if FLAGS.encoder_architecture:
        opts['encoder_architecture'] = FLAGS.encoder_architecture
    if FLAGS.decoder_architecture:
        opts['decoder_architecture'] = FLAGS.decoder_architecture
    if FLAGS.z_logvar_regularisation:
        if FLAGS.z_logvar_regularisation == "None":
            opts['z_logvar_regularisation'] = None
        else:
            opts['z_logvar_regularisation'] = FLAGS.z_logvar_regularisation
    if FLAGS.lambda_logvar_regularisation:
        opts['lambda_logvar_regularisation'] = FLAGS.lambda_logvar_regularisation
    if FLAGS.loss_regulariser:
        opts['loss_regulariser'] = FLAGS.loss_regulariser
    if FLAGS.beta:
        opts['beta'] = FLAGS.beta
    if FLAGS.plot_losses:
        opts['plot_losses'] = FLAGS.plot_losses    


    model = wae.Model(opts)
    model.train()

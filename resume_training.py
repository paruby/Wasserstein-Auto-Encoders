import os
import sys
import wae
import pickle

experiment_path = sys.argv[1]

if __name__ == "__main__":
    with open(experiment_path + "/opts.pickle", 'rb') as f:
        opts = pickle.load(f)
    model = wae.Model(opts, load=True)
    latest_checkpoint = tf.train.latest_checkpoint('checkpoints/')
    last_it = int(latest_checkpoint.split('-')[-1])
    model.train(last_it + 1)

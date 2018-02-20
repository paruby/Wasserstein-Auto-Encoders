import os
import sys
import wae
import pickle

experiment_path = sys.argv[1]

if __name__ == "__main__":
    opts = pickle.load(experiment_path + "/opts.pickle")
    model = wae.model(opts, load=True)
    

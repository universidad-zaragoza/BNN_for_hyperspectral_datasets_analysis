#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# Local imports
from lib import config
from lib.data import get_noisy_dataset
from lib.model import get_model
from lib.analysis import bayesian_predictions, analyse_entropy
from lib.plot import plot_combined_noise

# Testing all the images and generating all the noisy data can generate GPU
# memory errors.
# Try to comment this line if you have a big GPU. In any case, it will save the
# result of each dataset for future executions in case there are memory errors.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# PARAMETERS
# =============================================================================

def parse_args():
    """Analyses the received parameters and returns them organised.
    
    Takes the list of strings received at sys.argv and generates a
    namespace asigning them to objects.
    
    Returns
    -------
    out: namespace
        The namespace with the values of the received parameters asigned
        to objects.
    
    """
    # Generate the parameter analyser
    parser = ArgumentParser(description = __doc__,
                            formatter_class = RawDescriptionHelpFormatter)
    
    # Add arguments
    parser.add_argument("epochs",
                        type=int,
                        nargs=5,
                        help=("List of trained epochs. The order must be: BO, "
                              "IP, KSC, PU and SV."))
    parser.add_argument('-e', '--epoch',
                        type=int,
                        nargs=5,
                        help=("List of Selected epoch for testing. The order "
                              "must be: BO, IP, KSC, PU and SV. By default "
                              "uses `epochs` value."))    
    
    # Return the analysed parameters
    return parser.parse_args()

# PREDICT FUNCTIONS
# =============================================================================

def noise_predict(model, X_test, y_test, samples=100):
    
    # Bayesian stochastic passes
    predictions = bayesian_predictions(model, X_test, samples=samples)
    
    # Analyse entropy
    avg_H, _, _ = analyse_entropy(predictions, y_test)
    
    return avg_H

# MAIN
# =============================================================================

def main(epochs, epoch):
    
    # CONFIGURATION MACROS (extracted here as variables just for code clarity)
    # -------------------------------------------------------------------------
    
    # Input, output and dataset references
    d_path = config.DATA_PATH
    base_dir = config.LOG_DIR
    datasets = config.DATASETS
    output_dir = "Test"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Model parameters
    l1_n = config.LAYER1_NEURONS
    l2_n = config.LAYER2_NEURONS
    
    # Training parameters
    p_train = config.P_TRAIN
    learning_rate = config.LEARNING_RATE
    
    # Bayesian passes
    passes = config.BAYESIAN_PASSES
    
    # Plot parameters
    colors = config.COLORS
    w = config.PLOT_W
    h = config.PLOT_H
    
    # Plotting variables
    data = {}
    noises = np.arange(0.0, 0.61, 0.01)
    
    # FOR EVERY DATASET
    # -------------------------------------------------------------------------
    for name, dataset in datasets.items():
        
        # DATASET INFORMATION
        # ---------------------------------------------------------------------
        
        # Extract dataset classes and features
        num_classes = dataset['num_classes']
        num_features = dataset['num_features']
        
        # Get model dir
        model_dir = "{}_{}-{}model_{}train_{}ep_{}lr/epoch_{}".format(
                        name, l1_n, l2_n, p_train, epochs[name],
                        learning_rate, epoch[name])
        model_dir = os.path.join(base_dir, model_dir)
        if not os.path.isdir(model_dir):
            data[name] = []
            continue
        
        # Print dataset name and model dir
        print("\n# {}\n##########\n".format(name))
        print("MODEL DIR: {}\n".format(model_dir), flush=True)
        
        # GENERATE OR LOAD NOISY PREDICTIONS
        # ---------------------------------------------------------------------
        
        # If noisy predictions file already exists
        noise_file = os.path.join(model_dir, "test_noise.npy")
        if os.path.isfile(noise_file):
            
            # Load it
            noise_data = np.load(noise_file)
        
        else:
            
            # GET DATA
            # -----------------------------------------------------------------
            
            # Get noisy datasets
            (X_train, _,
             n_X_tests, n_y_test) = get_noisy_dataset(dataset, d_path,
                                                      p_train, noises)
            
            # LOAD MODEL
            # -----------------------------------------------------------------
            
            # Load model
            model = tf.keras.models.load_model(model_dir)
            
            # LAUNCH PREDICTIONS
            # -----------------------------------------------------------------
            
            # Launch predictions for every noisy dataset
            noise_data = [[] for i in range(num_classes + 1)]
            for n_X_test in n_X_tests:
                avg_H = noise_predict(model, n_X_test, n_y_test,
                                      samples=passes)
                noise_data = np.append(noise_data, avg_H[np.newaxis].T, 1)
            
            # Save result
            np.save(os.path.join(model_dir, "test_noise"), noise_data)
            
            # Liberate model
            del model
        
        # Add normalised average to data structure
        max_H = np.log(num_classes)
        data[name] = noise_data[-1]/max_H
        print("{}\t{}".format(name, data[name]))
    
    # Plot combined noise
    plot_combined_noise(output_dir, noises, data, w, h, colors)

if __name__ == "__main__":
    args = parse_args()
    if args.epoch is None:
        args.epoch = args.epochs
    epochs = {}
    epoch = {}
    for i, name in enumerate(["BO", "IP", "KSC", "PU", "SV"]):
        epochs[name] = args.epochs[i]
        epoch[name] = args.epoch[i]
    main(epochs, epoch)


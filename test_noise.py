#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf

# Local imports
import config
from data import get_noisy_dataset
from model import get_model
from analysis import bayesian_predictions, analyse_entropy
from plot import plot_uncertainty_with_noise

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

def main():
    
    # CONFIGURATION MACROS (extracted here as variables just for code clarity)
    # -------------------------------------------------------------------------
    
    # Input, output and dataset references
    d_path = config.DATA_PATH
    base_dir = config.LOG_DIR
    datasets = config.DATASETS
    
    # Model parameters
    l1_n = config.LAYER1_NEURONS
    l2_n = config.LAYER2_NEURONS
    
    # Training parameters
    p_train = config.P_TRAIN
    epochs = config.NUM_EPOCHS
    learning_rate = config.LEARNING_RATE
    
    # Bayesian passes
    passes = config.BAYESIAN_PASSES
    
    # Plot parameters
    colors = config.COLORS
    w = config.PLOT_W
    h = config.PLOT_H
    
    # FOR EVERY DATASET
    # -------------------------------------------------------------------------
    
    for name, dataset in datasets.items():
        
        # Extract dataset classes and features
        num_classes = dataset['num_classes']
        num_features = dataset['num_features']
        
        # Get output dir
        output_dir = "{}_{}-{}model_{}train_{}ep_{}lr".format(
                        name, l1_n, l2_n, p_train, epochs, learning_rate)
        output_dir = os.path.join(base_dir, output_dir)
        
        # Print dataset name and output dir
        print("\n# {}\n##########\n".format(name))
        print("OUTPUT DIR: {}\n".format(output_dir))
        
        # GET DATA
        # ---------------------------------------------------------------------
        
        # Get noisy datasets
        noises = np.arange(0.0, dataset['noise_stop'], dataset['noise_step'])
        X_train, _, n_X_tests, n_y_test = get_noisy_dataset(dataset, d_path,
                                                            p_train, noises)
        
        # LOAD MODEL
        # ---------------------------------------------------------------------
        
        # Get model
        dataset_size = len(X_train) + len(n_X_tests[0])
        model = get_model(dataset_size, num_features, num_classes, l1_n, l2_n,
                          learning_rate)
        
        # Load model parameters
        model = tf.keras.models.load_model(output_dir)
        
        # LAUNCH PREDICTIONS
        # ---------------------------------------------------------------------
        
        # Launch predictions for every noisy dataset
        noise_data = [[] for i in range(num_classes + 1)]
        for n_X_test in n_X_tests:
            avg_H = noise_predict(model, n_X_test, n_y_test, samples=passes)
            noise_data = np.append(noise_data, avg_H[np.newaxis].T, 1)
        
        # IMAGE-RELATED PLOTS
        # ---------------------------------------------------------------------
        
        # Plot uncertainty with noise
        plot_uncertainty_with_noise(output_dir, name, noises, noise_data, w, h,
                                    colors)

if __name__ == "__main__":
    main()


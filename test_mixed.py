#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf

# Local imports
import config
from data import get_dataset, get_mixed_dataset
from model import get_model
from analysis import bayesian_predictions, analyse_entropy
from plot import plot_mixed_uncertainty

# PREDICT FUNCTIONS
# =============================================================================

def predict(model, X_test, y_test, samples=100):
    
    # Bayesian stochastic passes
    predictions = bayesian_predictions(model, X_test, samples=samples)
    
    # Analyse entropy
    _, avg_Ep, _ = analyse_entropy(predictions, y_test)
    
    return avg_Ep

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
    
    # PLOTTING VARIABLES
    reliability_data = {}
    acc_data = {}
    px_data = {}
    
    # FOR EVERY DATASET
    # -------------------------------------------------------------------------
    
    for name, dataset in datasets.items():
        
        # Extract dataset classes and features
        num_classes = dataset['num_classes']
        num_features = dataset['num_features']
        class_a = dataset['mixed_class_A']
        class_b = dataset['mixed_class_B']
        
        # Get output dir and mixed output dir
        output_dir = "{}_{}-{}model_{}train_{}ep_{}lr".format(
                        name, l1_n, l2_n, p_train, epochs, learning_rate)
        mixed_output_dir = output_dir + "_{}-{}mixed".format(class_a, class_b)
        output_dir = os.path.join(base_dir, output_dir)
        mixed_output_dir = os.path.join(base_dir, mixed_output_dir)
        
        # Print dataset name and output dir
        print("\n# {}\n##########\n".format(name))
        print("OUTPUT DIR: {}\n".format(mixed_output_dir))
        
        # GET DATA
        # ---------------------------------------------------------------------
        
        # Get dataset
        X_train, _, X_test, y_test = get_dataset(dataset, d_path, p_train)
        
        # Get mixed dataset
        (m_X_train, _,
         m_X_test, m_y_test) = get_mixed_dataset(dataset, d_path, p_train,
                                                 class_a, class_b)
        
        # LOAD MODELS
        # ---------------------------------------------------------------------
        
        # Get model
        dataset_size = len(X_train) + len(X_test)
        model = get_model(dataset_size, num_features, num_classes, l1_n, l2_n,
                          learning_rate)
        
        # Load model parameters
        model = tf.keras.models.load_model(output_dir)
        
        # Get mixed model
        mixed_model = get_model(dataset_size, num_features, num_classes, l1_n,
                                l2_n, learning_rate)
        
        # Load mixed model parameters
        mixed_model = tf.keras.models.load_model(mixed_output_dir)
        
        # LAUNCH PREDICTIONS
        # ---------------------------------------------------------------------
        
        # Launch predictions
        avg_Ep = predict(model, X_test, y_test, samples=passes)
        
        # Launch mixed predictions
        m_avg_Ep = predict(mixed_model, m_X_test, m_y_test, samples=passes)
        
        # IMAGE-RELATED PLOTS
        # ---------------------------------------------------------------------
        
        # Plot class uncertainty
        data = [[avg_Ep[class_a], avg_Ep[class_b], avg_Ep[-1]],
                [m_avg_Ep[class_a], m_avg_Ep[class_b], m_avg_Ep[-1]]]
        plot_mixed_uncertainty(mixed_output_dir, name, data, class_a, class_b,
                               w, h, colors)

if __name__ == "__main__":
    main()


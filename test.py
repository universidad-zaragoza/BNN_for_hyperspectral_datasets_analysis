#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import os
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# Local imports
from .lib import config
from .lib.data import get_dataset
from .lib.model import get_model
from .lib.analysis import *
from .lib.plot import (plot_class_uncertainty, plot_reliability_diagram,
                      plot_accuracy_vs_uncertainty)

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

def predict(model, X_test, y_test, samples=100):
    
    # Bayesian stochastic passes
    predictions = bayesian_predictions(model, X_test, samples=samples)
    
    # Reliability Diagram
    rd_data = reliability_diagram(predictions, y_test)
    
    # Cross entropy and accuracy
    acc_data, px_data = accuracy_vs_uncertainty(predictions, y_test)
    
    # Analyse entropy
    _, avg_Ep, avg_H_Ep = analyse_entropy(predictions, y_test)
    
    return rd_data, acc_data, px_data, avg_Ep, avg_H_Ep

# MAIN
# =============================================================================

def main(epochs, epoch):
    
    # CONFIGURATION (extracted here as variables just for code clarity)
    # -------------------------------------------------------------------------
    
    # Input, output and dataset references
    d_path = config.DATA_PATH
    base_dir = config.MODELS_DIR
    datasets = config.DATASETS
    output_dir = config.TEST_DIR
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
    colours = config.COLOURS
    w = config.PLOT_W
    h = config.PLOT_H
    
    # Plotting variables
    reliability_data = {}
    acc_data = {}
    px_data = {}
    
    # FOR EVERY DATASET
    # -------------------------------------------------------------------------
    
    for name, dataset in datasets.items():
        
        # Extract dataset classes and features
        num_classes = dataset['num_classes']
        num_features = dataset['num_features']
        
        # Get model dir
        model_dir = "{}_{}-{}model_{}train_{}ep_{}lr/epoch_{}".format(
                        name, l1_n, l2_n, p_train, epochs[name], learning_rate,
                        epoch[name])
        model_dir = os.path.join(base_dir, model_dir)
        if not os.path.isdir(model_dir):
            reliability_data[name] = []
            acc_data[name] = []
            px_data[name] = []
            continue
        
        # Print dataset name and model dir
        print("\n# {}\n##########\n".format(name))
        print("MODEL DIR: {}\n".format(model_dir))
        
        # GET DATA
        # ---------------------------------------------------------------------
        
        # Get dataset
        X_train, _, X_test, y_test = get_dataset(dataset, d_path, p_train)
        
        # LOAD MODEL
        # ---------------------------------------------------------------------
        
        # Load model
        model = tf.keras.models.load_model(model_dir)
        
        # LAUNCH PREDICTIONS
        # ---------------------------------------------------------------------
        
        # Launch predictions
        (reliability_data[name],
         acc_data[name],
         px_data[name],
         avg_Ep, avg_H_Ep) = predict(model, X_test, y_test, samples=passes)
        
        # Liberate model
        del model
        
        # IMAGE-RELATED PLOTS
        # ---------------------------------------------------------------------
        
        # Plot class uncertainty
        plot_class_uncertainty(output_dir, name, epoch[name], avg_Ep, avg_H_Ep,
                               w, h, colours)
    
    # GROUPED PLOTS
    # -------------------------------------------------------------------------
    
    # Plot reliability diagram
    plot_reliability_diagram(output_dir, reliability_data, w, h, colours)
    
    # Plot accuracy vs uncertainty
    plot_accuracy_vs_uncertainty(output_dir, acc_data, px_data, w, h, colours)

if __name__ == "__main__":
    args = parse_args()
    if args.epoch is None:
        args.epoch = args.epochs
    epochs = {}
    epoch = {}
    for i, name in enumerate(config.DATASETS_LIST):
        epochs[name] = args.epochs[i]
        epoch[name] = args.epoch[i]
    main(epochs, epoch)

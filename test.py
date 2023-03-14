#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Testing module of the bnn4hi package

This module contains the main function to test the calibration of the
trained models generating the `reliability diagram`, test the accuracy
of the models with respect to the uncertainty of the predictions
generating the `uncertainty vs accuracy plot` and test the uncertainty
of each model, class by class, generating the `class uncertainty plot`
of each dataset.

This module can be imported as a part of the bnn4hi package, but it can
also be launched from command line, as a script. For that, use the `-h`
option to see the required arguments.
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
if '.' in __name__:
    
    # To run as a module
    from .lib import config
    from .lib.data import get_dataset
    from .lib.model import get_model
    from .lib.analysis import *
    from .lib.plot import (plot_class_uncertainty, plot_reliability_diagram,
                           plot_accuracy_vs_uncertainty)
else:
    
    # To run as a script
    from lib import config
    from lib.data import get_dataset
    from lib.model import get_model
    from lib.analysis import *
    from lib.plot import (plot_class_uncertainty, plot_reliability_diagram,
                          plot_accuracy_vs_uncertainty)
# PARAMETERS
# =============================================================================

def _parse_args(dataset_list):
    """Analyses the received parameters and returns them organised.
    
    Takes the list of strings received at sys.argv and generates a
    namespace assigning them to objects.
    
    Parameters
    ----------
    dataset_list : list of str
        List with the abbreviated names of the datasets to test. If
        `test.py` is launched as a script, the received parameters must
        correspond to the order of this list.
    
    Returns
    -------
    out : namespace
        The namespace with the values of the received parameters
        assigned to objects.
    """
    
    # Generate the parameter analyser
    parser = ArgumentParser(description = __doc__,
                            formatter_class = RawDescriptionHelpFormatter)
    
    # Add arguments
    parser.add_argument("epochs",
                        type=int,
                        nargs=len(dataset_list),
                        help=("List of the epoch of the selected checkpoint "
                              "for testing each model. The order must "
                              f"correspond to: {dataset_list}."))
    
    # Return the analysed parameters
    return parser.parse_args()

# PREDICT FUNCTIONS
# =============================================================================

def predict(model, X_test, y_test, samples=100):
    """Launches the bayesian predictions
    
    Launches the necessary predictions over `model` to collect the data
    to generate the `reliability diagram`, the `uncertainty vs accuracy
    plot` and the `class uncertainty` plot of the model.
    
    To generate the `reliability diagram` the predictions are divided
    into groups according to their predicted probability. To generate
    the `uncertainty vs accuracy` plot the predictions are divided into
    groups according to their uncertainty value. For that, it uses the
    default number of groups defined in the `reliability_diagram` and
    the `accuracy_vs_uncertainty` functions of `analysis.py`.
    
    Parameters
    ----------
    model : TensorFlow Keras Sequential
        The trained model.
    X_test : ndarray
        Testing data set.
    y_test : ndarray
        Testing data set labels.
    samples : int, optional (default: 100)
        Number of bayesian passes to perform.
    
    Returns
    -------
    rd_data : list of float
        List of the observed probabilities of each one of the predicted
        probability groups.
    acc_data : list of float
        List of the accuracies of each one of the uncertainty groups.
    px_data : list of float
        List of the percentage of pixels belonging to each one of the
        uncertainty groups.
    avg_Ep : ndarray
        List of the averages of the aleatoric uncertainty (Ep) of each
        class. The last position also contains the average of the
        entire image.
    avg_H_Ep : ndarray
        List of the averages of the epistemic uncertainty (H - Ep) of
        each class. The last position also contains the average of the
        entire image.
    """
    
    # Bayesian stochastic passes
    print(f"\nLaunching {samples} bayesian predictions")
    predictions = bayesian_predictions(model, X_test, samples=samples)
    
    # Reliability Diagram
    print("\nGenerating data for the `reliability diagram`", flush=True)
    rd_data = reliability_diagram(predictions, y_test)
    
    # Cross entropy and accuracy
    print("\nGenerating data for the `accuracy vs uncertainty` plot",
          flush=True)
    acc_data, px_data = accuracy_vs_uncertainty(predictions, y_test)
    
    # Analyse entropy
    print("\nGenerating data for the `class uncertainty` plot", flush=True)
    _, avg_Ep, avg_H_Ep = analyse_entropy(predictions, y_test)
    
    return rd_data, acc_data, px_data, avg_Ep, avg_H_Ep

# MAIN FUNCTION
# =============================================================================

def test(epochs):
    """Tests the trained bayesian models
    
    The plots are saved in the `TEST_DIR` defined in `config.py`.
    
    Parameters
    ----------
    epochs : dict
        Dict structure with the epochs of the selected checkpoint for
        testing each model. The keys must correspond to the abbreviated
        name of the dataset of each trained model.
    """
    
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
        model_dir = (f"{name}_{l1_n}-{l2_n}model_{p_train}train"
                     f"_{learning_rate}lr")
        model_dir = os.path.join(model_dir, f"epoch_{epochs[name]}")
        model_dir = os.path.join(base_dir, model_dir)
        if not os.path.isdir(model_dir):
            reliability_data[name] = []
            acc_data[name] = []
            px_data[name] = []
            continue
        
        # GET DATA
        # ---------------------------------------------------------------------
        
        # Get dataset
        _, _, X_test, y_test = get_dataset(dataset, d_path, p_train)
        
        # LOAD MODEL
        # ---------------------------------------------------------------------
        
        # Load trained model
        model = tf.keras.models.load_model(model_dir)
        
        # LAUNCH PREDICTIONS
        # ---------------------------------------------------------------------
        
        # Tests message
        print(f"\n### Starting {name} tests")
        print('#'*80)
        print(f"\nMODEL DIR: {model_dir}")
        
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
        plot_class_uncertainty(output_dir, name, epochs[name], avg_Ep,
                               avg_H_Ep, w, h)
    
    # End of tests message
    print("\n### Tests finished")
    print('#'*80, flush=True)
        
    # GROUPED PLOTS
    # -------------------------------------------------------------------------
    
    # Plot reliability diagram
    plot_reliability_diagram(output_dir, reliability_data, w, h, colours)
    
    # Plot accuracy vs uncertainty
    plot_accuracy_vs_uncertainty(output_dir, acc_data, px_data, w, h, colours)

if __name__ == "__main__":
    
    # Parse args
    dataset_list = config.DATASETS_LIST
    args = _parse_args(dataset_list)
    
    # Generate parameter structures for main function
    epochs = {}
    for i, name in enumerate(dataset_list):
        epochs[name] = args.epochs[i]
    
    # Launch main function
    test(epochs)

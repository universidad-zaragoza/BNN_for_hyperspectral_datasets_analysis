#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Map testing module of the bnn4hi package

This module contains the main function to generate the `uncertainty
map` of the bayesian trained model of a hyperspectral image.

It is generated along with an RGB representation of the hyperspectral
image, the ground truth and the `prediction map`. This visualisation is
intended to show the relationship between those images and the
importance of the information that can be extracted analysing the
uncertainty.

This module is prepared to be launched from command line, as a script,
but it can also be imported as a module from the bnn4hi package.
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import os
import datetime
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# Local imports
if '.' in __name__:
    
    # To run as a module
    from .lib import config
    from .lib.data import get_map, get_image
    from .lib.analysis import bayesian_predictions, map_prediction
    from .lib.plot import plot_maps

else:
    
    # To run as an script
    from lib import config
    from lib.data import get_map, get_image
    from lib.analysis import bayesian_predictions, map_prediction
    from lib.plot import plot_maps

# Some of the images are very big, so map testing generates GPU memory errors.
# Try to comment this line if you have a big GPU. In any case, it will save the
# result of each dataset for future executions in case there are memory errors.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# PARAMETERS
# =============================================================================

def parse_args(dataset_list):
    """Analyses the received parameters and returns them organised.
    
    Takes the list of strings received at sys.argv and generates a
    namespace assigning them to objects.
    
    Parameters
    ----------
    dataset_list : list of str
        List with the abbreviated names of the trained datasets.
    
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
    parser.add_argument("name",
                        choices=dataset_list,
                        help="Abbreviated name of the dataset.")
    parser.add_argument("epoch",
                        type=int,
                        help=("Number of trained epochs of the selected "
                              "checkpoint for testing."))
    
    # Return the analysed parameters
    return parser.parse_args()

# PREDICT FUNCTION
# =============================================================================

def map_predict(model, X, samples=100):
    """Launches the bayesian map predictions
    
    Launches the necessary predictions over `model` to collect the data
    to generate the `uncertainty map`.
    
    Parameters
    ----------
    model : TensorFlow Keras Sequential
        The trained model.
    X : ndarray
        The hyperspectral image pixels standardised.
    samples : int, optional (default: 100)
        Number of bayesian passes to perform.
    
    Returns
    -------
    pred_map : ndarray
        Array with the averages of the bayesian predictions.
    H_map : ndarray
        Array with the global uncertainty (H) values.
    """
    
    # Bayesian stochastic passes
    predictions = bayesian_predictions(model, X, samples=samples)
    
    # Map prediction
    return map_prediction(predictions)

# MAIN FUNCTION
# =============================================================================

def test_map(name, epoch):
    """Generates the `uncertainty map` of the trained bayesian model
    
    The final image is saved in the `TEST_DIR` defined in `config.py`.
    
    Parameters
    ----------
    name : str
        Abbreviated name of the dataset.
    epoch : int
        Number of epochs of the selected checkpoint for testing.
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
    
    # Maps colours
    colours = config.MAP_COLOURS
    gradients = config.MAP_GRADIENTS
    
    # DATASET INFORMATION
    # -------------------------------------------------------------------------
    
    dataset = datasets[name]
    
    # Extract dataset classes
    num_classes = dataset['num_classes']
    
    # Get model dir
    model_dir = "{}_{}-{}model_{}train_{}lr/epoch_{}".format(
                    name, l1_n, l2_n, p_train, learning_rate, epoch)
    model_dir = os.path.join(base_dir, model_dir)
    
    # Print dataset name and model dir
    print("\n# {}\n##########\n".format(name))
    print("MODEL DIR: {}\n{}\n".format(model_dir, datetime.datetime.now()))
    
    # GET DATA
    # -------------------------------------------------------------------------
    
    # Get dataset
    X, y, shape = get_map(dataset, d_path)
    
    # GENERATE OR LOAD MAP PREDICTIONS AND UNCERTAINTY
    # -------------------------------------------------------------------------
    
    # If prediction and uncertainty files already exist
    pred_map_file = os.path.join(model_dir, "pred_map.npy")
    H_map_file = os.path.join(model_dir, "H_map.npy")
    if os.path.isfile(pred_map_file) and os.path.isfile(H_map_file):
        
        # Load them
        pred_map = np.load(pred_map_file)
        H_map = np.load(H_map_file)
    
    else:
        
        # Load model parameters
        model = tf.keras.models.load_model(model_dir)
        
        # Launch predictions
        pred_map, H_map = map_predict(model, X, samples=passes)
        
        # Save prediction and uncertainty files
        np.save(os.path.join(model_dir, "pred_map"), pred_map)
        np.save(os.path.join(model_dir, "H_map"), H_map)
        
        # Liberate model
        del model
    
    # PLOT MAPS
    # -------------------------------------------------------------------------
    
    # Get image and wavelengths
    img, _ = get_image(dataset, d_path)
    wl = dataset['wl']
    
    # Plot
    plot_maps(output_dir, name, shape, num_classes, wl, img, y, pred_map,
              H_map, colours, gradients)

if __name__ == "__main__":
    
    # Parse args
    dataset_list = config.DATASETS_LIST
    args = parse_args(dataset_list)
    
    # Launch main function
    test_map(args.name, args.epoch)

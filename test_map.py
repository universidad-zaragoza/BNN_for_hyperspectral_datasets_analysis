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
import datetime
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# Local imports
from .lib import config
from .lib.data import get_map, get_image
from .lib.analysis import bayesian_predictions, map_prediction
from .lib.plot import plot_maps

# Some of the images are very big, so map testing generates GPU memory errors.
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
    parser.add_argument("name",
                        choices=["BO", "IP", "KSC", "PU", "SV"],
                        help="Abbreviated name of the dataset.")
    parser.add_argument("epochs",
                        type=int,
                        help="Trained epochs.")
    parser.add_argument('-e', '--epoch',
                        type=int,
                        help=("Selected epoch for testing. By default uses "
                              "`epochs` value."))
    parser.add_argument('-l', '--legend',
                        action='store_true',
                        help="Flag to activate the generation of legends.")

    # Return the analysed parameters
    return parser.parse_args()

# PREDICT FUNCTION
# =============================================================================

def predict(model, X, samples=100):
    
    # Bayesian stochastic passes
    predictions = bayesian_predictions(model, X, samples=samples)
    
    # Map prediction
    return map_prediction(predictions)

# MAIN
# =============================================================================

def main(name, epochs, epoch, legend):
    
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
    model_str = "{}-{}model".format(l1_n, l2_n,)
    model_dir = "{}_{}_{}train_{}ep_{}lr/epoch_{}".format(
                    name, model_str, p_train, epochs, learning_rate, epoch)
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
        pred_map, H_map = predict(model, X, samples=passes)
        
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
    args = parse_args()
    if args.epoch is None:
        args.epoch = args.epochs
    main(args.name, args.epochs, args.epoch, args.legend)

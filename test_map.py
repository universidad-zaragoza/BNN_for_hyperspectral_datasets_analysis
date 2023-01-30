#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import math
import datetime
import numpy as np
import tensorflow as tf
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as grid
from spectral import imshow
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# Local imports
from lib import config
from lib.data import get_map, get_image, _load_image
from lib.analysis import bayesian_predictions, map_prediction
from lib.HSI2RGB import HSI2RGB

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

# MAP FUNCTIONS
# =============================================================================

def uncertainty_to_map(uncertainty, num_classes, slots=10, max_H=0):
    if max_H == 0:
        max_H = math.log(num_classes)
    u_map = np.zeros(uncertainty.shape, dtype="int")
    ranges = np.linspace(0.0, max_H, num=slots+1)
    labels = ["0.0-{:.2f}".format(ranges[1])]
    slot = 1
    start = ranges[1]
    for end in ranges[2:]:
        u_map[(start <= uncertainty) & (uncertainty <= end)] = slot
        labels.append("{:.2f}-{:.2f}".format(start, end))
        start = end
        slot +=1
    return u_map, labels

def map_to_img(prediction, shape, colors, metric=None, th=0.0, bg=(0, 0, 0)):
    img_shape = (shape[0], shape[1], 3)
    if metric is not None:
        return np.reshape([colors[int(p)] if m < th else bg
                           for p, m in zip(prediction, metric)], img_shape)
    else:
        return np.reshape([colors[int(p)] for p in prediction], img_shape)

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
    
    # Maps colors
    colors_rgb = config.COLORS_RGB
    colors_int_rgb = config.COLORS_INT_RGB
    gradients_rgb = config.GRADIENTS_RGB
    gradients_int_rgb = config.GRADIENTS_INT_RGB
    
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
    
    # GENERATE OR LOAD PREDICTIONS AND UNCERTAINTY
    # -------------------------------------------------------------------------
    
    # If prediction and uncertainty files already exist
    pred_map_file = os.path.join(model_dir, "pred_map.npy")
    H_map_file = os.path.join(model_dir, "H_map.npy")
    if os.path.isfile(pred_map_file) and os.path.isfile(H_map_file):
        
        # Load them
        pred_map = np.load(pred_map_file)
        H_map = np.load(H_map_file)
    
    else:
        
        # LOAD MODEL
        # ---------------------------------------------------------------------
        
        # Load model parameters
        model = tf.keras.models.load_model(model_dir)
        
        # Launch predictions
        pred_map, H_map, _, _ = predict(model, X, samples=passes)
        
        # Save prediction and uncertainty files
        np.save(os.path.join(model_dir, "pred_map"), pred_map)
        np.save(os.path.join(model_dir, "H_map"), H_map)
        
        # Liberate model
        del model
    
    # PREPARE FIGURE
    # -------------------------------------------------------------------------
    
    fig = plt.figure(frameon=False)
    dpi = 96
    fig.set_size_inches(shape[1]/dpi, shape[0]/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # RGB IMAGE GENERATION
    # Using HSI2RGB algorithm from paper:
    #     M. Magnusson, J. Sigurdsson, S. E. Armansson, M. O. Ulfarsson,
    #     H. Deborah and J. R. Sveinsson, "Creating RGB Images from
    #     Hyperspectral Images Using a Color Matching Function," IGARSS 2020 -
    #     2020 IEEE International Geoscience and Remote Sensing Symposium,
    #     2020, pp. 2045-2048, doi: 10.1109/IGARSS39084.2020.9323397.
    # HSI2RGB code from:
    #     https://github.com/JakobSig/HSI2RGB
    # -------------------------------------------------------------------------
    
    # Get image and wavelengths
    img, _ = get_image(dataset, d_path)
    wl = dataset['wl']
    
    # Create RGB image (D65 illuminant and 0.002 threshold)
    RGB_img = HSI2RGB(wl, img, shape[0], shape[1], 65, 0.002)
    
    # Generate and save image
    file = "{}_RGB.pdf".format(name)
    ax.imshow(RGB_img)
    plt.savefig(os.path.join(output_dir, file), bbox_inches='tight')
    
    # GROUND TRUTH GENERATION
    # -------------------------------------------------------------------------
    
    gt = map_to_img(y, shape, [(0, 0, 0)] + colors_int_rgb[:num_classes])
    file = "{}_gt.pdf".format(name)
    ax.imshow(gt)
    handles = [mpatches.Patch(color=colors_rgb[i], label="class {}".format(i))
               for i in range(num_classes)]
    if legend:
        plt.legend(handles=handles, title="Classes", loc="upper left",
                   bbox_to_anchor=(1, 1))
    plt.savefig(os.path.join(output_dir, file), bbox_inches='tight')
    
    # PREDICTION MAP GENERATION
    # -------------------------------------------------------------------------
    
    pred_H_img = map_to_img(pred_map, shape, colors_int_rgb[:num_classes])
    file = "{}_{}_pred_map.pdf".format(name, epoch)
    ax.imshow(pred_H_img)
    handles = [mpatches.Patch(color=colors_rgb[i], label="class {}".format(i))
               for i in range(num_classes)]
    if legend:
        plt.legend(handles=handles, title="Classes", loc="upper left",
                   bbox_to_anchor=(1, 1))
    plt.savefig(os.path.join(output_dir, file), bbox_inches='tight')
    
    # UNCERTAINTY MAP GENERATION
    # -------------------------------------------------------------------------
    
    slots = 15
    u_map, labels = uncertainty_to_map(H_map, num_classes, slots=slots,
                                       max_H=1.5)
    H_img = map_to_img(u_map, shape, gradients_int_rgb[:slots])
    file = "{}_{}_H_map.pdf".format(name, epoch)
    ax.imshow(H_img)
    handles = [mpatches.Patch(color=gradients_rgb[i], label=labels[i])
               for i in range(slots)]
    if legend:
        plt.legend(handles=handles, title="Uncertainty", loc="upper left",
                   bbox_to_anchor=(1, 1))
    plt.savefig(os.path.join(output_dir, file), bbox_inches='tight')

if __name__ == "__main__":
    args = parse_args()
    if args.epoch is None:
        args.epoch = args.epochs
    main(args.name, args.epochs, args.epoch, args.legend)


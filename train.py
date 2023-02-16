#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Training module of the BNN4HI package

"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import os
import sys
import time
import tensorflow as tf
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# Local imports
from .lib import config
from .lib.data import get_dataset, get_mixed_dataset
from .lib.model import get_model

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
                        help="Total number of epochs.")
    parser.add_argument("period",
                        type=int,
                        help="Checkpoints and information period.")
    parser.add_argument('-m', '--mix_classes',
                        action='store_true',
                        help="Flag to activate mixed classes training.")
    
    # Return the analysed parameters
    return parser.parse_args()

# PRINT CALLBACK FUNCTION
# =============================================================================

class PrintCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, print_epoch=1000, losses_avg_no=100):
        self.print_epoch = print_epoch
        self.losses_avg_no = losses_avg_no
    
    def print_loss_acc(self, logs, time, last=False):
        loss = sum(self.losses[-self.losses_avg_no:])/self.losses_avg_no
        if last:
            print("\n--- TRAIN END AT EPOCH {} ---".format(self.epoch))
            print("TRAINING TIME: {} seconds".format(time))
        else:
            print("CURRENT TIME: {} seconds".format(time))
        print("Epoch loss ({}): {}".format(self.epoch, loss))
        print("Accuracy: {}".format(logs.get('val_accuracy')), flush=True)
    
    def on_train_begin(self, logs={}):
        self.losses = []
        self.epoch = 0
        self.start_time = time.time()
    
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
    
    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        if self.epoch % self.print_epoch == 0:
            current_time = time.time() - self.start_time
            self.print_loss_acc(logs, current_time)
    
    def on_train_end(self, logs={}):
        total_time = time.time() - self.start_time
        self.print_loss_acc(logs, total_time, last=True)

# MAIN
# =============================================================================

def main(name, epochs, period, mix_classes):
    
    # CONFIGURATION (extracted here as variables just for code clarity)
    # -------------------------------------------------------------------------
    
    # Input, output and dataset references
    d_path = config.DATA_PATH
    base_output_dir = config.MODELS_DIR
    datasets = config.DATASETS
    
    # Model parameters
    l1_n = config.LAYER1_NEURONS
    l2_n = config.LAYER2_NEURONS
    
    # Training parameters
    p_train = config.P_TRAIN
    learning_rate = config.LEARNING_RATE
    
    # DATASET INFORMATION
    # -------------------------------------------------------------------------
    
    dataset = datasets[name]
    
    # Extract dataset classes and features
    num_classes = dataset['num_classes']
    num_features = dataset['num_features']
    
    # Generate output dir
    output_dir = "{}_{}-{}model_{}train_{}ep_{}lr".format(
                    name, l1_n, l2_n, p_train, epochs, learning_rate)
    if mix_classes:
        class_a = dataset['mixed_class_A']
        class_b = dataset['mixed_class_B']
        output_dir += "_{}-{}mixed".format(class_a, class_b)
    output_dir = os.path.join(base_output_dir, output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Print dataset name and output dir
    print("\n# {}\n##########\n".format(name))
    print("OUTPUT DIR: {}\n".format(output_dir), flush=True)
    
    # GET DATA
    # -------------------------------------------------------------------------
    
    # Get dataset
    if mix_classes:
        (X_train, y_train,
         X_test, y_test) = get_mixed_dataset(dataset, d_path, p_train,
                                             class_a, class_b)
    else:
        (X_train, y_train,
         X_test, y_test) = get_dataset(dataset, d_path, p_train)
    
    # TRAIN MODEL
    # -------------------------------------------------------------------------
    
    # Get model (if already trained, continue for finetuning)
    trained = [int(d.split("_")[1]) for d in os.listdir(output_dir)
               if "_" in d]
    if trained:
        initial_epoch = max(trained)
        last_file = os.path.join(output_dir, "epoch_{}".format(initial_epoch))
        model = tf.keras.models.load_model(last_file)
    else:
        initial_epoch = 0
        dataset_size = len(X_train) + len(X_test)
        model = get_model(dataset_size, num_features, num_classes, l1_n, l2_n,
                          learning_rate)
    
    # PRINT CALLBACK
    print_callback = PrintCallback(period, max(1, period//10))
    
    # CHECKPOINT CALLBACK
    file = os.path.join(output_dir, "epoch_{epoch}")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file,
                                                    monitor='val_accuracy',
                                                    mode='max',
                                                    save_best_only=False,
                                                    period=period)
    
    # Training
    model.fit(X_train,
              tf.one_hot(y_train, num_classes),
              initial_epoch=initial_epoch,
              epochs=epochs, 
              verbose=0,
              use_multiprocessing=True,
              callbacks=[print_callback, checkpoint],
              validation_split=0.1,
              validation_freq=25)
    
    # Save model
    model.save("{}/final".format(output_dir))

if __name__ == "__main__":
    args = parse_args()
    main(args.name, args.epochs, args.period, args.mix_classes)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import tensorflow as tf

# Local imports
import config
from data import get_dataset, get_mixed_dataset
from model import get_model

# CALLBACK FUNCTION
# =============================================================================

class PrintCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, print_epoch=1000, losses_avg_no=100):
        self.print_epoch = print_epoch
        self.losses_avg_no = losses_avg_no
    
    def print_loss_acc(self, logs, last=False, time=None):
        loss = sum(self.losses[-self.losses_avg_no:])/self.losses_avg_no
        if last:
            print("\n--- TRAIN END AT EPOCH {} ---".format(self.epoch))
            print("TRAINING TIME: {} seconds".format(time))
        print("Epoch loss ({}): {}".format(self.epoch, loss))
        print("Accuracy: {}".format(logs.get('val_accuracy')))
    
    def on_train_begin(self, logs={}):
        self.losses = []
        self.epoch = 0
        self.start_time = time.time()
    
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
    
    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        if self.epoch % self.print_epoch == 0:
            self.print_loss_acc(logs)
    
    def on_train_end(self, logs={}):
        total_time = time.time() - self.start_time
        self.print_loss_acc(logs, last=True, time=total_time)

# MAIN
# =============================================================================

def main():
    
    # CONFIGURATIONS (extracted here as variables just for code clarity)
    # -------------------------------------------------------------------------
    
    # Input, output and dataset references
    d_path = config.DATA_PATH
    base_output_dir = config.LOG_DIR
    datasets = config.DATASETS
    
    # Model parameters
    l1_n = config.LAYER1_NEURONS
    l2_n = config.LAYER2_NEURONS
    
    # Training parameters
    p_train = config.P_TRAIN
    epochs = config.NUM_EPOCHS
    learning_rate = config.LEARNING_RATE
    
    # Callback parameters
    print_epoch = config.PRINT_EPOCH
    losses = config.LOSSES_AVG_NO
    
    # Get mix classes parameter
    mix_classes = "-m" in sys.argv
    
    # FOR EVERY DATASET
    # -------------------------------------------------------------------------
    
    for name, dataset in datasets.items():
        
        # Extract dataset classes and features
        num_classes = dataset['num_classes']
        num_features = dataset['num_features']
        class_a = dataset['mixed_class_A']
        class_b = dataset['mixed_class_B']
        
        # Generate output dir
        output_dir = "{}_{}-{}model_{}train_{}ep_{}lr".format(
                        name, l1_n, l2_n, p_train, epochs, learning_rate)
        if mix_classes:
            output_dir += "_{}-{}mixed".format(class_a, class_b)
        output_dir = os.path.join(base_output_dir, output_dir)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
        # Print dataset name and output dir
        print("\n# {}\n##########\n".format(name))
        print("OUTPUT DIR: {}\n".format(output_dir))
        
        # GET DATA
        # ---------------------------------------------------------------------
        
        # Get dataset
        if mix_classes:
            (X_train, y_train,
             X_test, y_test) = get_mixed_dataset(dataset, d_path, p_train,
                                                 class_a, class_b)
        else:
            (X_train, y_train,
             X_test, y_test) = get_dataset(dataset, d_path, p_train)
        
        # TRAIN MODEL
        # ---------------------------------------------------------------------
        
        # Get model
        dataset_size = len(X_train) + len(X_test)
        model = get_model(dataset_size, num_features, num_classes, l1_n, l2_n,
                          learning_rate)
        
        # Training
        history = model.fit(X_train,
                            tf.one_hot(y_train, num_classes),
                            epochs=epochs, 
                            verbose=0, 
                            use_multiprocessing=True, 
                            callbacks=[PrintCallback(print_epoch, losses)],
                            validation_split=0.1,
                            validation_freq=25)
        
        # Save model
        model.save(output_dir)

if __name__ == "__main__":
    main()


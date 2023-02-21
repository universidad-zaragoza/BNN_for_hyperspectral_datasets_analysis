#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Model module of the bnn4hi package

This module defines the bayesian model used to train.
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import os
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as dist

# MODEL FUNCTION
# =============================================================================

def get_model(dataset_size, num_features, num_classes, l1_n, l2_n,
              learning_rate):
    """Generates the bayesian model
    
    Parameters
    ----------
    dataset_size : int
        Number of pixels of the dataset.
    num_features : int
        Number of features of each pixel.
    num_classes : int
        Number of classes of the dataset.
    l1_n : int
        Number of neurons of the first hidden layer.
    l2_n : int
        Number of neurons of the second hidden layer
    learning_rate : float
        Initial learning rate.
    
    Returns
    -------
    model : TensorFlow Keras Sequential
        Bayesian model ready to receive and train hyperspectral data.
    """
    
    # Generate and compile model
    tf.keras.backend.clear_session()
    kd_function = (lambda q, p, _: dist.kl_divergence(q, p) / dataset_size)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(num_features,), name="input"),
        tfp.layers.DenseFlipout(l1_n, kernel_divergence_fn=kd_function,
                                activation=tf.nn.relu, name="dense_tfp_1"),
        tfp.layers.DenseFlipout(l2_n, kernel_divergence_fn=kd_function,
                                activation=tf.nn.relu, name="dense_tfp_2"),
        tfp.layers.DenseFlipout(num_classes,
                                kernel_divergence_fn=kd_function,
                                activation=tf.nn.softmax, name="output"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

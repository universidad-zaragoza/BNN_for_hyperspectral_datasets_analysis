#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as dist

# MODEL FUNCTION
# =============================================================================

def get_model(dataset_size, num_features, num_classes, l1_n, l2_n,
              learning_rate):
    
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


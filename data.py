#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy import io
from numpy.random import randint as rand
from sklearn.model_selection import train_test_split

# DATA FUNCTIONS
# =============================================================================

def _load_image(image_info, data_path):
    """Loads the image and the ground truth from a `mat` file.
    
    If the file is not present in the `data_path` directory, downloads
    the file from the `image_info` url.
    
    Parameters
    ----------
    image_info: dict
        Dict structure with information of the image. Described below.
    data_path: String
        Absolute path of the hyperspectral images directory.
    
    Returns
    -------
    out: NumPy array, NumPy array
        The image and the ground truth data.
    
    """
    # Image name
    image_name = image_info['key']
    
    # Filenames
    input_file = os.path.join(data_path, image_info['file'])
    label_file = os.path.join(data_path, image_info['file_gt'])
    
    # LOAD IMAGE
    try:
        
        # Load image file
        X = io.loadmat(input_file)[image_name]
    
    except:
        
        # Download image file
        os.system("wget {} -O {}".format(image_info['url'], input_file))
	    
        # Load image file
        X = io.loadmat(input_file)[image_name]
    
    # LOAD GROUND TRUTH
    try:
        
        # Load ground truth file
        y = io.loadmat(label_file)[image_info['key_gt']]
    
    except:
        
        # Download ground truth file
        os.system("wget {} -O {}".format(image_info['url_gt'], label_file))
	    
        # Load ground truth file
        y = io.loadmat(label_file)[image_info['key_gt']]
    
    return X, y

def _standardise(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

def _preprocess(X, y, standardisation=False, only_labeled=True):
    
    # Reshape them to ignore spatiality
    X = X.reshape(-1, X.shape[2])
    y = y.reshape(-1)
    
    if only_labeled:
        
        # Keep only labeled pixels
        X = X[y > 0, :]
        y = y[y > 0]
    
    # Rename clases to ordered integers from 0
    for new_class_num, old_class_num in enumerate(np.unique(y)):
        y[y == old_class_num] = new_class_num
    
    if standardisation:
        X = _standardise(X)
    
    return X, y

# GET DATASET FUNCTION
# =============================================================================

def get_dataset(dataset, data_path, p_train, seed=35):
    
    # Load image
    X, y = _load_image(dataset, data_path)
    
    # Preprocess
    X, y = _preprocess(X, y, standardisation=True)
    
    # Separate into train, val and test data sets
    p_test = 1 - p_train
    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, test_size=p_test,
                                         random_state=seed, stratify=y)
    
    return X_train, y_train, X_test, y_test

# NOISY DATASET FUNCTIONS
# =============================================================================

def _generic_noise(X_test, p_noise):
    """Generates random value variations for every feature of X_test"""
    return rand(-2**15, 2**15 - 1, size=X_test.shape, dtype='int16')*p_noise

def get_noisy_dataset(dataset, data_path, p_train, noises, seed=35):
    
    # Load image
    X, y = _load_image(dataset, data_path)
    
    # Noise preprocessing settings (to standardise after generating the noise)
    X_mean = X.reshape(-1, X.shape[2]).mean(axis=0)
    X_std = X.reshape(-1, X.shape[2]).std(axis=0)
    
    # Preprocess
    X, y = _preprocess(X, y, standardisation=False)
    
    # Separate into train, val and test data sets
    p_test = 1 - p_train
    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, test_size=p_test,
                                         random_state=seed, stratify=y)
    
    noisy_X_tests = []
    for noise in noises:
        
        # Add noise to X_test
        noisy_X_tests.append(X_test + _generic_noise(X_test, noise))
    
    # Standardise X_train and each noisy_X_tests
    X_train = (X_train - X_mean) / X_std
    for n, X_test in enumerate(noisy_X_tests):
        noisy_X_tests[n] = (X_test - X_mean) / X_std
    
    # Return train and test sets
    return X_train, y_train, noisy_X_tests, y_test

# MIXED DATASET FUNCTIONS
# =============================================================================

def _mix_classes(y_train, class_a, class_b):
    """Mixes the labels between two classes"""
    
    # Get the indices of the pixels from both classes
    index = (y_train == class_a) | (y_train == class_b)
    
    # Estract and shuffle their values
    values = y_train[index]
    np.random.shuffle(values)
    
    # Modify the original values with the new ones
    y_train[index] = values

def get_mixed_dataset(dataset, data_path, p_train, class_a, class_b, seed=35):
    
    # Get dataset
    X_train, y_train, X_test, y_test = get_dataset(dataset, data_path, p_train,
                                                   seed=seed)
    
    # Mix the labels between two classes
    _mix_classes(y_train, class_a, class_b)
    
    return X_train, y_train, X_test, y_test


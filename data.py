#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy import io
from sklearn.model_selection import train_test_split

# DATA FUNCTIONS
# =============================================================================

def load_image(image_info, data_path):
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

def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

def preprocess(X, y, standardization=False, only_labeled=True):
    
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
    
    if standardization:
        X = standardize(X)
    
    return X, y

def getDataset(dataset, data_path, p_train, seed=35):
    
    # Load image
    X, y = load_image(dataset, data_path)
    
    # Preprocess
    X, y = preprocess(X, y, standardization=True)
    
    # Separate into train, val and test data sets
    p_test = 1 - p_train
    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, test_size=p_test,
                                         random_state=seed, stratify=y)
    
    return X_train, y_train, X_test, y_test


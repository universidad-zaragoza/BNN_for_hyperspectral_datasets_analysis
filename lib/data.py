#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data module of the bnn4hi package

The functions of this module are used to load, preprocess and organise
data from hyperspectral datasets.
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import os
import numpy as np
from scipy import io
from numpy.random import randint as rand
from sklearn.model_selection import train_test_split

# DATA FUNCTIONS
# =============================================================================

def _load_image(image_info, data_path):
    """Loads the image and the ground truth from a `mat` file
    
    If the file is not present in the `data_path` directory, downloads
    the file from the `image_info` url.
    
    Parameters
    ----------
    image_info : dict
        Dict structure with information of the image. Described in the
        config module of bnn4hi package.
    data_path : str
        Path of the hyperspectral images and ground truth files. It can
        be an absolute path or relative from the execution path.
    
    Returns
    -------
    X : ndarray
        Hyperspectral image.
    y : ndarray
        Ground truth.
    """
    
    # Image name
    image_name = image_info['key']
    
    # Generate data path if it does not exist
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    
    # Filenames
    input_file = os.path.join(data_path, image_info['file'])
    label_file = os.path.join(data_path, image_info['file_gt'])
    
    # LOAD IMAGE
    try:
        
        # Load image file
        X = io.loadmat(input_file)[image_name]
    
    except:
        
        # Download image file
        os.system(f"wget {image_info['url']} -O {input_file}")
	    
        # Load image file
        X = io.loadmat(input_file)[image_name]
    
    # LOAD GROUND TRUTH
    try:
        
        # Load ground truth file
        y = io.loadmat(label_file)[image_info['key_gt']]
    
    except:
        
        # Download ground truth file
        os.system(f"wget {image_info['url_gt']} -O {label_file}")
	    
        # Load ground truth file
        y = io.loadmat(label_file)[image_info['key_gt']]
    
    return X, y

def _standardise(X):
    """Standardises a set of hyperspectral pixels
    
    Parameters
    ----------
    X : ndarray
        Set of hyperspectral pixels.
    
    Returns
    -------
    X_standardised : ndarray
        The received set of pixels standardised.
    """
    
    return (X - X.mean(axis=0))/X.std(axis=0)

def _normalise(X):
    """Normalises a set of hyperspectral pixels
    
    Parameters
    ----------
    X : ndarray
        Set of hyperspectral pixels.
    
    Returns
    -------
    X_normalised : ndarray
        The received set of pixels normalised.
    """
    
    X -= X.min()
    return X/X.max()

def _preprocess(X, y, standardisation=False, only_labelled=True):
    """Preprocesses the hyperspectral image and ground truth data
    
    Parameters
    ----------
    X : ndarray
        Hyperspectral image.
    y : ndarray
        Ground truth of `X`.
    standardistion : bool, optional (default: False)
        Flag to activate standardisation.
    only_labelled : bool, optional (default: True)
        Flag to remove unlabelled pixels.
    
    Returns
    -------
    X : ndarray
        Preprocessed data of the hyperspectral image.
    y : ndarray
        Preprocessed data of ground truth.
    """
    
    # Reshape them to ignore spatiality
    X = X.reshape(-1, X.shape[2])
    y = y.reshape(-1)
    
    if only_labelled:
        
        # Keep only labelled pixels
        X = X[y > 0, :]
        y = y[y > 0]
        
        # Rename clases to ordered integers from 0
        for new_class_num, old_class_num in enumerate(np.unique(y)):
            y[y == old_class_num] = new_class_num
        
        # Regular standardisation
        if standardisation:
            X = _standardise(X)
    
    # Standardise only using labelled pixels for `mean` and `std`
    elif standardisation:
        m = X[y > 0, :].mean(axis=0)
        s = X[y > 0, :].std(axis=0)
        X = (X - m)/s
    
    return X, y

# GET DATASET FUNCTION
# =============================================================================

def get_dataset(dataset, data_path, p_train, seed=35):
    """Returns the preprocessed training and testing data and labels
    
    Parameters
    ----------
    dataset : dict
        Dict structure with information of the image. Described in the
        config module of bnn4hi package.
    data_path : str
        Path of the datasets. It can be an absolute path or relative
        from the execution path.
    p_train : float
        Represents, from 0.0 to 1.0, the proportion of the dataset that
        will be used for the training set.
    seed : int, optional (default: 35)
        Random seed used to shuffle the data. The same seed will
        produce the same distribution of pixels between train and test
        sets. The default value (35) is just there for reproducibility
        purposes, as it is the used seed in the paper `Bayesian Neural
        Networks to Analyze Hyperspectral Datasets Using Uncertainty
        Metrics`.
    
    Returns
    -------
    X_train : ndarray
        Training data set.
    y_train : ndarray
        Training data set labels.
    X_test : ndarray
        Testing data set.
    y_test : ndarray
        Testing data set labels.
    """
    
    # Load image
    X, y = _load_image(dataset, data_path)
    
    # Preprocess
    X, y = _preprocess(X, y, standardisation=True)
    
    # Separate into train and test data sets
    p_test = 1 - p_train
    (X_train, X_test,
     y_train, y_test) = train_test_split(X, y, test_size=p_test,
                                         random_state=seed, stratify=y)
    
    return X_train, y_train, X_test, y_test

# NOISY DATASET FUNCTIONS
# =============================================================================

def _generic_noise(shape, p_noise):
    """Generates an array of `int16` random value variations
    
    Parameters
    ----------
    shape : tuple of ints
        The shape of the expected array.
    p_noise : float
        Represents, from 0.0 to 1.0, the noise factor. The closest to
        1.0 will produce the higher noise values.
    
    Returns
    -------
    variations : ndarray
        Array of random variations within the values range of `int16`
        and proportional to `p_noise`.
    """
    
    return rand(-2**15, 2**15 - 1, size=shape, dtype='int16')*p_noise

def get_noisy_dataset(dataset, data_path, p_train, noises, seed=35):
    """Returns train set and several test sets with increasing noise
    
    Parameters
    ----------
    dataset: dict
        Dict structure with information of the image. Described in the
        config module of bnn4hi package.
    data_path: str
        Path of the datasets. It can be an absolute path or relative
        from the execution path.
    p_train: float
        Represents, from 0.0 to 1.0, the proportion of the dataset that
        will be used for the training set.
    noises : array_like of floats
        Each value represents, from 0.0 to 1.0, one noise factor. The
        closest to 1.0 will produce the higher noise values. A testing
        set will be generated for each received noise.
    seed: int, optional (default: 35)
        Random seed used to shuffle the data. The same seed will
        produce the same distribution of pixels between train and test
        sets. The default value (35) is just there for reproducibility
        purposes, as it is the used seed in the paper `Bayesian Neural
        Networks to Analyze Hyperspectral Datasets Using Uncertainty
        Metrics`.
    
    Returns
    -------
    X_train : ndarray
        Training data set.
    y_train : ndarray
        Training data set labels.
    noisy_X_tests : ndarray
        One testing data set per received noise.
    y_test : ndarray
        Testing data set labels. It is the same for every testing set.
    """
    
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
        
        # Add noise to `X_test` and append it to `noisy_X_tests`
        noisy_X_tests.append(X_test + _generic_noise(X_test.shape, noise))
    
    # Standardise `X_train` and each `noisy_X_tests`
    X_train = (X_train - X_mean)/X_std
    for n, X_test in enumerate(noisy_X_tests):
        noisy_X_tests[n] = (X_test - X_mean)/X_std
    
    # Return train and test sets
    return X_train, y_train, noisy_X_tests, y_test

# MIXED DATASET FUNCTIONS
# =============================================================================

def _mix_classes(y_train, class_a, class_b):
    """Mixes the labels between two classes
    
    It does not return, the `y_train` array is modified in-place.
    
    Parameters
    ----------
    y_train : ndarray
        Training data set labels to be modified.
    class_a : int
        Number of the first class to be mixed.
    class_b : int
        Number of the second class to be mixed.
    """
    
    # Get the indices of the pixels from both classes
    index = (y_train == class_a) | (y_train == class_b)
    
    # Estract and shuffle their values
    values = y_train[index]
    np.random.shuffle(values)
    
    # Modify the original values with the new ones
    y_train[index] = values

def get_mixed_dataset(dataset, data_path, p_train, class_a, class_b, seed=35):
    """Returns the datasets with mixed classes on the training set
    
    Parameters
    ----------
    dataset : dict
        Dict structure with information of the image. Described in the
        config module of bnn4hi package.
    data_path : str
        Path of the datasets. It can be an absolute path or relative
        from the execution path.
    p_train : float
        Represents, from 0.0 to 1.0, the proportion of the dataset that
        will be used for the training set.
    class_a : int
        Number of the first class to be mixed.
    class_b : int
        Number of the second class to be mixed.
    seed : int, optional (default: 35)
        Random seed used to shuffle the data. The same seed will
        produce the same distribution of pixels between train and test
        sets. The default value (35) is just there for reproducibility
        purposes, as it is the used seed in the paper `Bayesian Neural
        Networks to Analyze Hyperspectral Datasets Using Uncertainty
        Metrics`.
    
    Returns
    -------
    X_train : ndarray
        Training data set.
    y_train : ndarray
        Training data set labels with mixed labels for `class_a` and
        `class_b`.
    X_test : ndarray
        Testing data set.
    y_test : ndarray
        Testing data set labels.
    """
    
    # Get dataset
    X_train, y_train, X_test, y_test = get_dataset(dataset, data_path, p_train,
                                                   seed=seed)
    
    # Mix the labels between two classes
    _mix_classes(y_train, class_a, class_b)
    
    return X_train, y_train, X_test, y_test

# MAP FUNCTIONS
# =============================================================================

def get_map(dataset, data_path):
    """Returns all the pixels and labels of the image preprocessed
    
    Parameters
    ----------
    dataset : dict
        Dict structure with information of the image. Described in the
        config module of bnn4hi package.
    data_path : str
        Path of the datasets. It can be an absolute path or relative
        from the execution path.
    
    Returns
    -------
    X : ndarray
        Hyperspectral image pixels standardised.
    y : ndarray
        Ground truth.
    shape : tuple of ints
        Original shape to reconstruct the image (without channels, just
        height and width).
    """
    
    # Load image
    X, y = _load_image(dataset, data_path)
    shape = y.shape
    
    # Preprocess
    X, y = _preprocess(X, y, standardisation=True, only_labelled=False)
    
    return X, y, shape

# IMAGE FUNCTIONS (FOR RGB REPRESENTATION)
# =============================================================================

def get_image(dataset, data_path):
    """Returns the image prepared for the RGB representation algorithm
    
    Parameters
    ----------
    dataset : dict
        Dict structure with information of the image. Described in the
        config module of bnn4hi package.
    data_path : str
        Path of the datasets. It can be an absolute path or relative
        from the execution path.
    
    Returns
    -------
    X : ndarray
        Hyperspectral image pixels normalised.
    shape : tuple of ints
        Original shape to reconstruct the image.
    """
    
    # Load image
    X, y = _load_image(dataset, data_path)
    shape = X.shape
    
    # Preprocess
    X, _ = _preprocess(X, y, standardisation=False, only_labelled=False)
    
    # Normalise
    X = _normalise(X)
    
    return X, shape

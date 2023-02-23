#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Analysis module of the bnn4hi package

The functions of this module are used to generate and analyse bayesian
predictions.
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import sys
import math
import numpy as np

# UNCERTAINTY FUNCTIONS
#     Global uncertainty (H) corresponds to predictive entropy
#     Aleatoric uncertainty (Ep) corresponds to expected entropy
#     Epistemic uncertainty corresponds to H - Ep subtraction
# =============================================================================

def _predictive_entropy(predictions):
    """Calculates the predictive entropy of `predictions`
    
    The predictive entropy corresponds to the global uncertainty (H).
    The correspondent equation can be found in the paper `Bayesian
    Neural Networks to Analyze Hyperspectral Datasets Using Uncertainty
    Metrics`.
    
    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    
    Returns
    -------
    pred_h : ndarray
        Predictive entropy, i.e. global uncertainty, of `predictions`
    """
    
    # Get number of pixels and classes
    _, num_pixels, num_classes = predictions.shape
    
    # Application of the predictive entropy equation
    entropy = np.zeros(num_pixels)
    for p in range(num_pixels):
        for c in range(num_classes):
            avg = np.mean(predictions[..., p, c])
            if avg == 0.0:
                avg = sys.float_info.min
            entropy[p] += avg * math.log(avg)
    
    return -1 * entropy

def _expected_entropy(predictions):
    """Calculates the expected entropy of `predictions`
    
    The expected entropy corresponds to the aleatoric uncertainty (Ep).
    The correspondent equation can be found in the paper `Bayesian
    Neural Networks to Analyze Hyperspectral Datasets Using Uncertainty
    Metrics`.
    
    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    
    Returns
    -------
    pred_ep : ndarray
        Expected entropy, i.e. aleatoric uncertainty, of `predictions`
    """
    
    # Get number of bayesian passes, pixels and classes
    num_tests, num_pixels, num_classes = predictions.shape
    
    # Application of the expected entropy equation
    entropy = np.zeros(num_pixels)
    for p in range(num_pixels):
        for t in range(num_tests):
            class_sum = 0
            for c in range(num_classes):
                val = predictions[t][p][c]
                if val == 0.0:
                    val = sys.float_info.min
                class_sum += val * math.log(val)
            entropy[p] -= class_sum
    
    return entropy / num_tests

# ANALYSIS FUNCTIONS
# =============================================================================

def reliability_diagram(predictions, y_test, num_groups=10):
    """Generates the `reliability diagram` data
    
    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    y_test : ndarray
        Testing data set labels.
    num_groups : int, optional (default: 10)
        Number of groups in which the prediction will be divided
        according to their predicted probability.
    
    Returns
    -------
    result : list of float
        List of the observed probabilities of each one of the predicted
        probability groups.
    """
    
    # Get number of classes
    num_classes = predictions.shape[2]
    
    # Calculate the bayesian samples average
    prediction = np.mean(predictions, axis=0)
    
    # Labels to one-hot encoding
    labels = np.zeros((len(y_test), num_classes))
    labels[np.arange(len(y_test)), y_test] = 1
    
    # Probability groups to divide predictions
    p_groups = np.linspace(0.0, 1.0, num_groups + 1)
    p_groups[-1] += 0.1 # To include the last value
    
    result = []
    for i in range(num_groups):
        
        # Calculate the average of each group
        group = labels[(prediction >= p_groups[i]) &
                       (prediction < p_groups[i + 1])]
        result.append(group.sum() / len(group))
    
    return result

def accuracy_vs_uncertainty(predictions, y_test, H_limit=1.5, num_groups=15):
    """Generates the `accuracy vs uncertainty` data
    
    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    y_test : ndarray
        Testing data set labels.
    H_limit : float, optional (default: 1.5)
        The max value of the range of uncertainty.
    num_groups : int, optional (default: 15)
        Number of groups in which the prediction will be divided
        according to their uncertainty.
    
    Returns
    -------
    H_acc : list of float
        List of the accuracies of each one of the uncertainty groups.
    p_pixels : list of float
        List of the percentage of pixels belonging to each one of the
        uncertainty groups.
    """
    
    # Get predictive entropy
    test_H = _predictive_entropy(predictions)
    
    # Generate a boolean map of hits
    test_ok = np.mean(predictions, axis=0).argmax(axis=1) == y_test
    
    # Uncertainty groups to divide predictions
    H_groups = np.linspace(0.0, H_limit, num_groups + 1)
    
    H_acc = []
    p_pixels = []
    for i in range(num_groups):
        
        # Calculate the average and percentage of pixels of each group
        group = test_ok[(test_H >= H_groups[i]) & (test_H < H_groups[i + 1])]
        p_pixels.append(len(group)/len(y_test))
        H_acc.append(group.sum()/len(group))
    
    return H_acc, p_pixels

def analyse_entropy(predictions, y_test):
    """Calculates the average uncertainty values by class
    
    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    y_test : ndarray
        Testing data set labels.
    
    Returns
    -------
    class_H_avg : ndarray
        List of the averages of the global uncertainty (H) of each
        class. The last position also contains the average of the
        entire image.
    class_Ep_avg : ndarray
        List of the averages of the aleatoric uncertainty (Ep) of each
        class. The last position also contains the average of the
        entire image.
    class_H_Ep_avg : ndarray
        List of the averages of the epistemic uncertainty (H - Ep) of
        each class. The last position also contains the average of the
        entire image.
    """
    
    # Get the uncertainty values
    model_H = _predictive_entropy(predictions)
    model_Ep = _expected_entropy(predictions)
    model_H_Ep = model_H - model_Ep
    
    # Structures for the averages
    num_classes = predictions.shape[2]
    class_H = np.zeros(num_classes + 1)
    class_Ep = np.zeros(num_classes + 1)
    class_H_Ep = np.zeros(num_classes + 1)
    class_px = np.zeros(num_classes + 1, dtype='int')
    
    for px, (H, Ep, H_Ep, label) in enumerate(zip(model_H, model_Ep,
                                                  model_H_Ep, y_test)):
        
        # Label as integer
        label = int(label)
        
        # Accumulate uncertainty values by class
        class_H[label] += H
        class_Ep[label] += Ep
        class_H_Ep[label] += H_Ep
        
        # Count pixels for class average
        class_px[label] += 1
        
        # Accumulate for every class
        class_H[-1] += H
        class_Ep[-1] += Ep
        class_H_Ep[-1] += H_Ep
        
        # Count pixels for global average
        class_px[-1] += 1
    
    # Return averages
    return class_H/class_px, class_Ep/class_px, class_H_Ep/class_px

def map_prediction(predictions):
    """Returns the bayesian predictions and global uncertainties (H)
    
    This function is implemented to facilitate all the data required
    for the maps comparisons.
    
    Parameters
    ----------
    predictions : ndarray
        Array with the bayesian predictions.
    
    Returns
    -------
    pred_map : ndarray
        Array with the averages of the bayesian predictions.
    test_H : ndarray
        Array with the global uncertainty (H) values.
    """
    
    # Calculate the bayesian samples average prediction
    pred_map = np.mean(predictions, axis=0).argmax(axis=1)
    
    # Get the global uncertainty values
    test_H = _predictive_entropy(predictions)
    
    return pred_map, test_H

# PREDICTIONS FUNCTION
# =============================================================================

def bayesian_predictions(model, X_test, samples=100):
    """Generates bayesian predictions
    
    Parameters
    ----------
    model : TensorFlow Keras Sequential
        Trained bayesian model.
    X_test : ndarray
        Testing data set.
    samples : int, optional (default: 100)
        Number of bayesian passes to perform.
    
    Returns
    -------
    predictions : ndarray
        Array with the bayesian predictions.
    """
    
    # Bayesian stochastic passes
    predictions = []
    for i in range(samples):
        
        # Progress bar
        status = int(78*len(predictions)/samples)
        print('[' + '='*(status) + ' '*(78 - status) + ']', end="\r",
              flush=True)
        
        # Launch prediction
        prediction = model.predict(X_test, verbose=0)
        predictions.append(prediction)
    
    # End of progress bar
    print('[' + '='*78 + ']', flush=True)
    
    return np.array(predictions)

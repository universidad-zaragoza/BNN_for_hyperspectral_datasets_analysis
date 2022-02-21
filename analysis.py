#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import math
import numpy as np

# UNCERTAINTY FUNCTIONS
# =============================================================================

# H --> PREDICTIVE (ALEATORIC + EPISTEMIC)
def _predictive_entropy(prediction):
    
    _, num_pixels, num_classes = prediction.shape
    
    entropy = np.zeros(num_pixels)
    for p in range(num_pixels):
        for c in range(num_classes):
            avg = np.mean(prediction[..., p, c])
            if avg == 0.0:
                avg = sys.float_info.min
            entropy[p] += avg * math.log(avg)
    
    return -1 * entropy

# Ep --> ALEATORIC
def _expected_entropy(prediction):
    
    num_tests, num_pixels, num_classes = prediction.shape
    
    entropy = np.zeros(num_pixels)
    for p in range(num_pixels):
        for t in range(num_tests):
            class_sum = 0
            for c in range(num_classes):
                val = prediction[t][p][c]
                if val == 0.0:
                    val = sys.float_info.min
                class_sum += val * math.log(val)
            entropy[p] -= class_sum
    
    return entropy / num_tests

# ANALYSIS FUNCTIONS
# =============================================================================

def reliability_diagram(predictions, y_test, num_groups=10):
    
    num_classes = predictions.shape[2]
    
    prediction = np.mean(predictions, axis=0) # Bayesian samples average
    labels = np.zeros((len(y_test), num_classes))
    labels[np.arange(len(y_test)), y_test] = 1 # Labels to one-hot
    
    p_groups = np.linspace(0.0, 1.0, num_groups + 1)
    p_groups[-1] += 0.1 # For including the last value
    
    result = []
    for i in range(num_groups):
        group_avg = labels[(prediction >= p_groups[i]) & (prediction < p_groups[i + 1])]
        result.append(group_avg.sum() / len(group_avg))
    
    return result

def accuracy_vs_uncertainty(predictions, y_test, num_groups=15):
    
    test_H = _predictive_entropy(predictions)
    test_ok = np.mean(predictions, axis=0).argmax(axis=1) == y_test
    
    H_groups = np.linspace(0.0, 1.5, num_groups + 1)
    
    H_acc = []
    p_pixels = []
    for i in range(num_groups):
        group = test_ok[(test_H >= H_groups[i]) & (test_H < H_groups[i + 1])]
        p_pixels.append(len(group)/len(y_test))
        H_acc.append(group.sum() / len(group))
    
    return H_acc, p_pixels

def analyse_entropy(prediction, y_test):
    
    model_H = _predictive_entropy(prediction)
    model_Ep = _expected_entropy(prediction)
    model_H_Ep = model_H - model_Ep
    
    num_classes = prediction.shape[2]
    class_H = np.zeros(num_classes + 1)
    class_Ep = np.zeros(num_classes + 1)
    class_H_Ep = np.zeros(num_classes + 1)
    class_px = np.zeros(num_classes + 1, dtype='int')
    
    for px, (H, Ep, H_Ep, label) in enumerate(zip(model_H, model_Ep,
                                                  model_H_Ep, y_test)):
        
        label = int(label)
        
        class_H[label] += H
        class_Ep[label] += Ep
        class_H_Ep[label] += H_Ep
        class_px[label] += 1
        
        class_H[-1] += H
        class_Ep[-1] += Ep
        class_H_Ep[-1] += H_Ep
        class_px[-1] += 1
    
    return class_H/class_px, class_Ep/class_px, class_H_Ep/class_px

# PREDICTIONS FUNCTION
# =============================================================================

def bayesian_predictions(model, X_test, samples=100):
    
    # Bayesian stochastic passes
    predictions = []
    for i in range(samples):
        prediction = model.predict(X_test)
        predictions.append(prediction)
    
    return np.array(predictions)


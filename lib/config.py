#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Config module of the bnn4hi package

This module provides macros with data information, training and testing
parameters and plotting configurations.
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import numpy as np

# CONFIGURATION GLOBALS
# =============================================================================

# Uncomment if there are GPU memory errors during training
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Uncomment for CPU execution. Only recommended if there are GPU memory
# errors during testing
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Input and output directories
DATA_PATH = "./Data"
MODELS_DIR = "./Models"
TEST_DIR = "./Test"

# DATA INFORMATION
# =============================================================================

# Datasets
#     The classes to mix when training with mixed classes have been
#     selected for having enough and similar number of labelled pixels.
#     Wavelengths have been selected according to the sensor and the
#     characteristics of each image for a better RGB representation.
url_base = "http://www.ehu.es/ccwintco/uploads"
DATASETS = {
    "BO": {
        'file': "Botswana.mat",
        'file_gt': "Botswana_gt.mat",
        'key': "Botswana",
        'key_gt': "Botswana_gt",
        'url': url_base + "/7/72/Botswana.mat",
        'url_gt': url_base + "/5/58/Botswana_gt.mat",
        'num_classes': 14,
        'num_features': 145,
        'mixed_class_A': 4,
        'mixed_class_B': 5,
        # NASA EO-1 wavelengths (for RGB representation)
        'wl' : np.linspace(400, 2500, num=242).take([
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
            38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
            101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
            114, 115, 116, 117, 118, 133, 134, 135, 136, 137, 138, 139, 140,
            141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
            154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 186, 187, 188,
            189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201,
            202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214,
            215, 216, 217, 218, 219]).tolist()
    },
    "IP": {
        'file': "Indian_pines_corrected.mat",
        'file_gt': "Indian_pines_gt.mat",
        'key': "indian_pines_corrected",
        'key_gt': "indian_pines_gt",
        'url': url_base + "/6/67/Indian_pines_corrected.mat",
        'url_gt': url_base + "/c/c4/Indian_pines_gt.mat",
        'num_classes': 16,
        'num_features': 200,
        'mixed_class_A': 2,
        'mixed_class_B': 5,
        # AVIRIS wavelengths for IP (for RGB representation)
        'wl' : np.linspace(400, 2500, num=224).take([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
            87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
            108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
            121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
            134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
            147, 148, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
            174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186,
            187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
            200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
            213, 214, 215, 216, 217, 218, 220, 221, 223]).tolist()
    },
    "KSC": {
        'file': "KSC.mat",
        'file_gt': "KSC_gt.mat",
        'key': "KSC",
        'key_gt': "KSC_gt",
        'url': url_base + "/2/26/KSC.mat",
        'url_gt': url_base + "/a/a6/KSC_gt.mat",
        'num_classes': 13,
        'num_features': 176,
        'mixed_class_A': 8,
        'mixed_class_B': 11,
        # AVIRIS wavelengths for KSC (for RGB representation)
        'wl' : np.linspace(400, 2500, num=224).take([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
            87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
            111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
            137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 167,
            168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
            181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193,
            194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206,
            207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 220,
            221, 222]).tolist()
    },
    "PU": {
        'file': "PaviaU.mat",
        'file_gt': "PaviaU_gt.mat",
        'key': "paviaU",
        'key_gt': "paviaU_gt",
        'url': url_base + "/e/ee/PaviaU.mat",
        'url_gt': url_base + "/5/50/PaviaU_gt.mat",
        'num_classes': 9,
        'num_features': 103,
        'mixed_class_A': 3,
        'mixed_class_B': 7,
        # ROSIS wavelengths (for RGB representation)
        'wl' : np.linspace(430, 860, num=115).tolist()
    },
     "SV": {
        'file': "Salinas_corrected.mat",
        'file_gt': "Salinas_gt.mat",
        'key': "salinas_corrected",
        'key_gt': "salinas_gt",
        'url': url_base + "/a/a3/Salinas_corrected.mat",
        'url_gt': url_base + "/f/fa/Salinas_gt.mat",
        'num_classes': 16,
        'num_features': 204,
        'mixed_class_A': 1,
        'mixed_class_B': 6,
        # AVIRIS wavelengths for SV (for RGB representation)
        'wl' : np.linspace(400, 2500, num=224).take([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
            70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
            87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
            108, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
            137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
            150, 151, 152, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
            177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
            190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
            203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
            216, 217, 218, 219, 220, 221, 222]).tolist()
    }
}

# Sorted datasets list
#     It is used to manage parameters order, as dict items are ordered
#     or not depending on python version.
DATASETS_LIST = ["BO", "IP", "KSC", "PU", "SV"]

# TRAINING AND TESTING PARAMETERS
# =============================================================================

# Model parameters
LAYER1_NEURONS = 32
LAYER2_NEURONS = 16

# Training parameters
P_TRAIN = 0.5
LEARNING_RATE = 1.0e-2

# Bayesian passes
BAYESIAN_PASSES = 100

# List of noises for noise tests
NOISES = np.arange(0.0, 0.61, 0.01)

# PLOTTING CONFIGURATIONS
# =============================================================================

# Plots size
PLOT_W = 7
PLOT_H = 4

# Plots colours
COLOURS = {"BO": "#2B4162",
           "IP": "#FA9F42",
           "KSC": "#0B6E4F",
           "PU": "#721817",
           "SV": "#D496A7"}

# Maps colours
# 99% accessibility colours (https://sashamaps.net/docs/resources/20-colors/)
MAP_COLOURS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (70, 240, 240), (240, 50, 230), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200),
    (128, 0, 0), (170, 255, 195), (0, 0, 128), (128, 128, 128)]
MAP_GRADIENTS = [
    (77, 230, 54), (135, 229, 53), (193, 229, 52), (229, 206, 51),
    (228, 146, 50), (228, 86, 49), (228, 48, 71), (227, 47, 130),
    (227, 46, 189), (204, 45, 227), (143, 44, 226), (81, 43, 226),
    (42, 64, 226), (41, 125, 225), (40, 185, 225), (39, 225, 202)]

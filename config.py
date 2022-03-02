#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# CONFIGURATION GLOBALS
# =============================================================================

# Uncomment if there are GPU memory errors during training
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Uncomment for CPU execution
# Only recommended if there are GPU memory errors during testing
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Input and output directories
DATA_PATH = "./Data"
LOG_DIR = "./Models"

# Image files
#     The noise step and stop values are empirically selected for better
#     visualisation. For other datasets it will be necessary to adjust them.
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
       'noise_stop': 0.13,
       'noise_step': 0.01,
       'mixed_class_A': 4,
       'mixed_class_B': 5
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
       'noise_stop': 0.13,
       'noise_step': 0.01,
       'mixed_class_A': 2,
       'mixed_class_B': 5
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
       'noise_stop': 0.13,
       'noise_step': 0.01,
       'mixed_class_A': 8,
       'mixed_class_B': 11
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
       'noise_stop': 0.62,
       'noise_step': 0.02,
       'mixed_class_A': 3,
       'mixed_class_B': 7
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
        'noise_stop': 0.13,
        'noise_step': 0.01,
        'mixed_class_A': 1,
        'mixed_class_B': 6
    }
}

# Model parameters
LAYER1_NEURONS = 32
LAYER2_NEURONS = 16

# Training parameters
P_TRAIN = 0.5
NUM_EPOCHS = 20000
LEARNING_RATE = 1.0e-2

# Callback parameters
PRINT_EPOCH = 200
LOSSES_AVG_NO = 10

# Bayesian passes
BAYESIAN_PASSES = 100

# Plot parameters
COLORS = {"BO": "#2B4162",
          "IP": "#FA9F42",
          "KSC": "#0B6E4F",
          "PU": "#721817",
          "SV": "#D496A7"}
PLOT_W = 7
PLOT_H = 4


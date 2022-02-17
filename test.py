#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as dist
from scipy.stats import mode

import config
from data import getDataset

# PLOT FUNCTIONS
# =============================================================================

def plot_reliability_diagram(output_dir, data, w, h, colors, num_groups=10):
    
    # Generate x axis labels and center of range (for optimal calibration)
    p_groups = np.linspace(0.0, 1.0, num_groups + 1)
    center = (p_groups[1] - p_groups[0]) / 2
    optimal = (p_groups + center)[:-1]
    if num_groups <= 10:
        labels = ["{:.1f}-{:.1f}".format(p_groups[i], p_groups[i + 1])
                  for i in range(num_groups)]
    else:
        labels = ["{:.2f}-{:.2f}".format(p_groups[i], p_groups[i + 1])
                  for i in range(num_groups)]
    
    # Xticks
    xticks = np.arange(len(labels))
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
    for img_name in colors.keys():
        ax.plot(xticks[:len(data[img_name])], data[img_name], label=img_name,
                color=colors[img_name])
    ax.plot(xticks, optimal, label="Optimal calibration", color='black',
            linestyle='dashed')
    
    # Axes label
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    
    # Y axis limit
    ax.set_ylim((0, 1))
    
    # X axis limit
    ax.set_xlim((xticks[0], xticks[-1]))
    
    # Rotate X axis labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Grid
    ax.grid(axis='y')
    
    # Legend
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
    
    # Save
    plt.savefig(os.path.join(output_dir, "reliability_diagram.pdf"),
                bbox_inches='tight')

def plot_accuracy_vs_uncertainty(output_dir, acc_data, px_data, w, h, colors,
                                 H_limit=1.5, num_groups=15):
    
    # Labels
    H_groups = np.linspace(0.0, H_limit, num_groups + 1)
    labels = ["{:.2f}-{:.2f}".format(H_groups[i], H_groups[i + 1])
              for i in range(num_groups)]
    
    # Xticks
    xticks = np.arange(len(labels))
    
    # Yticks
    yticks = np.arange(0, 1.1, 0.1)
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
    for img_name in colors.keys():
        ax.plot(xticks[:len(acc_data[img_name])], acc_data[img_name],
                label="{} acc.".format(img_name), color=colors[img_name],
                zorder=3)
        ax.bar(xticks[:len(px_data[img_name])], px_data[img_name],
               label="{} px %".format(img_name), color=colors[img_name],
               alpha=0.18, zorder=2)
        ax.bar(xticks[:len(px_data[img_name])],
               [-0.007 for i in px_data[img_name]], bottom=px_data[img_name],
               color=colors[img_name], zorder=3)
    
    # Axes label
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Pixels % and accuracy")
    
    # Y axis limit
    ax.set_ylim((0, 1))
    
    # X axis limit
    ax.set_xlim((xticks[0] - 0.5, xticks[-1] + 0.5))
    
    # Y axis minors
    ax.set_yticks(yticks, minor=True)
    
    # Rotate X axis labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Grid
    ax.grid(axis='y', zorder=1)
    ax.grid(axis='y', which='minor', linestyle='dashed', zorder=1)
    
    # Legend handles and labels and new order
    lg_handles, lg_labels = plt.gca().get_legend_handles_labels()
    
    # Manual legend to adjust the handles and place labels in a new order
    order = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
    ax.add_artist(ax.legend([lg_handles[idx] for idx in order],
                            [lg_labels[idx] for idx in order],
                            loc='upper center', ncol=5,
                            bbox_to_anchor=(0.46, 1.2)))
    
    # Manualy added handles upper lines (to resamble the bars)
    ax.add_artist(ax.legend([lg_handles[0]], [""], framealpha=0,
                            handlelength=1.8, loc='upper center',
                            bbox_to_anchor=(-0.0555, 1.146)))
    ax.add_artist(ax.legend([lg_handles[1]], [""], framealpha=0,
                            handlelength=1.8, loc='upper center',
                            bbox_to_anchor=(0.177, 1.146)))
    ax.add_artist(ax.legend([lg_handles[2]], [""], framealpha=0,
                            handlelength=1.8, loc='upper center',
                            bbox_to_anchor=(0.3947, 1.146)))
    ax.add_artist(ax.legend([lg_handles[3]], [""], framealpha=0,
                            handlelength=1.8, loc='upper center',
                            bbox_to_anchor=(0.6406, 1.146)))
    ax.add_artist(ax.legend([lg_handles[4]], [""], framealpha=0,
                            handlelength=1.8, loc='upper center',
                            bbox_to_anchor=(0.8698, 1.146)))
    
    # Save
    plt.savefig(os.path.join(output_dir, "accuracy_vs_uncertainty.pdf"),
                bbox_inches='tight')

def plot_class_uncertainty(output_dir, name, avg_Ep, avg_H_Ep, w, h, colors):
    
    # Xticks
    xticks = np.arange(len(avg_Ep))
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
    ax.bar(xticks, avg_Ep, label="Ep", color=colors["BO"], zorder=3)
    ax.bar(xticks, avg_H_Ep, bottom=avg_Ep, label="H - Ep",
           color=colors["KSC"], zorder=3)
    
    # Highlight avg border
    ax.bar(xticks[-1], avg_Ep[-1] + avg_H_Ep[-1], zorder=2,
           edgecolor=colors["IP"], linewidth=4)
    
    # Axes label
    ax.set_xlabel("{} classes".format(name))
    
    # X axis labels
    ax.set_xticks(xticks)
    xlabels = np.append(xticks[:-1], ["AVG"])
    ax.set_xticklabels(xlabels)
    
    # Grid
    ax.grid(axis='y', zorder=1)
    
    # Legend
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
    
    # Save
    file_name = "{}_class_uncertainty.pdf".format(name)
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')

# UNCERTAINTY FUNCTIONS
# =============================================================================

# H --> PREDICTIVE (ALEATORIC + EPISTEMIC)
def predictive_entropy(prediction):
    
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
def expected_entropy(prediction):
    
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

def entropy_vs_accuracy(predictions, y_test, num_groups=15):
    
    test_H = predictive_entropy(predictions)
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
    
    model_H = predictive_entropy(prediction)
    model_Ep = expected_entropy(prediction)
    model_H_Ep = model_H - model_Ep
    
    num_classes = prediction.shape[2]
    class_Ep = np.zeros(num_classes + 1)
    class_H_Ep = np.zeros(num_classes + 1)
    class_px = np.zeros(num_classes + 1, dtype='int')
    
    for px, (Ep, H_Ep, label) in enumerate(zip(model_Ep, model_H_Ep, y_test)):
        
        label = int(label)
        
        class_Ep[label] += Ep
        class_H_Ep[label] += H_Ep
        class_px[label] += 1
        
        class_Ep[-1] += Ep
        class_H_Ep[-1] += H_Ep
        class_px[-1] += 1
    
    return class_Ep/class_px, class_H_Ep/class_px

# PREDICT FUNCTION
# =============================================================================

def predict(output_dir, model, X_test, y_test, samples=100):
    
    # Bayesian stochastic passes
    class_predictions = []
    for i in range(samples):
        prediction = model.predict(X_test)
        class_predictions.append(prediction)
    class_predictions = np.array(class_predictions)
    
    # Reliability Diagram
#    rd_data = reliability_diagram(class_predictions, y_test)
    rd_data = []
    # Cross entropy and accuracy
#    acc_data, px_data = entropy_vs_accuracy(class_predictions, y_test)
    acc_data = []
    px_data = []
    
    # Analyse entropy
    avg_Ep, avg_H_Ep = analyse_entropy(class_predictions, y_test)
    
    return rd_data, acc_data, px_data, avg_Ep, avg_H_Ep

# MAIN
# =============================================================================

def main(argv):
    
    # CONFIGURATIONS (extracted here as variables just for code clarity)
    
    # Input, output and dataset references
    d_path = config.DATA_PATH
    base_dir = config.LOG_DIR
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
    
    # Bayesian passes
    passes = config.BAYESIAN_PASSES
    
    # Plot parameters
    colors = config.COLORS
    w = config.PLOT_W
    h = config.PLOT_H
    
    # PLOTTING VARIABLES
    reliability_data = {}
    acc_data = {}
    px_data = {}
    
    for name, dataset in datasets.items():
        
        # Extract dataset classes and features
        num_classes = dataset['num_classes']
        num_features = dataset['num_features']
        
        # Generate output dir
        output_dir = "{}_{}-{}model_{}train_{}ep_{}lr".format(
                        name, l1_n, l2_n, p_train, epochs, learning_rate)
        output_dir = os.path.join(base_dir, output_dir)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
        # Print dataset name and output dir
        print("\n# {}\n##########\n".format(name))
        print("OUTPUT DIR: {}\n".format(output_dir))
        
        # Get dataset
        X_train, y_train, X_test, y_test = getDataset(dataset, d_path, p_train)
        
        # Generate model
        tf.keras.backend.clear_session()
        dataset_size = len(X_train) + len(X_test)
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
        
        # Load model
        model = tf.keras.models.load_model(output_dir)
        
        # Launch redictions
        (reliability_data[name],
         acc_data[name],
         px_data[name],
         avg_Ep, avg_H_Ep) = predict(output_dir, model, X_test, y_test,
                                     samples=passes)
        
        # Plot class uncertainty
        plot_class_uncertainty(output_dir, name, avg_Ep, avg_H_Ep, w, h,
                               colors)
    
    # Plot reliability diagram
    plot_reliability_diagram(base_dir, reliability_data, w, h, colors)
    
    # Plot accuracy vs uncertainty
    plot_accuracy_vs_uncertainty(base_dir, acc_data, px_data, w, h, colors)

if __name__ == "__main__":
    main(sys.argv[1:])


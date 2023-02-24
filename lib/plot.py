#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot module of the bnn4hi package

The functions of this module are used to generate plots using the
results of the analysed bayesian predictions.
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from .HSI2RGB import HSI2RGB

# MAP FUNCTIONS
# =============================================================================

def _uncertainty_to_map(uncertainty, num_classes, slots=10, max_H=0):
    """Groups the uncertainty values received into uncertainty groups
    
    Parameters
    ----------
    uncertainty : ndarray
        Array with the uncertainty values.
    num_classes : int
        Number of classes of the dataset.
    slots : int, optional (default: 10)
        Number of groups to divide uncertainty map values.
    max_H : float, optional (default: 0)
        The max value of the range of uncertainty for the uncertainty
        map. The `0` value will use the logarithm of `num_classes` as
        it is the theoretical maximum value of the uncertainty.
    
    Returns
    -------
    u_map : ndarray
        List with the uncertainty group corresponding to each
        uncertainty value received.
    labels : list of strings
        List of the labels for plotting the `u_map` value groups.
    """
    
    # Actualise `max_H` in case of the default value
    if max_H == 0:
        max_H = math.log(num_classes)
    
    # Prepare output structures and ranges
    u_map = np.zeros(uncertainty.shape, dtype="int")
    ranges = np.linspace(0.0, max_H, num=slots+1)
    labels = ["0.0-{:.2f}".format(ranges[1])]
    
    # Populate the output structures
    slot = 1
    start = ranges[1]
    for end in ranges[2:]:
        
        # Fill with the slot number and actualise labels
        u_map[(start <= uncertainty) & (uncertainty <= end)] = slot
        labels.append("{:.2f}-{:.2f}".format(start, end))
        
        # For next iteration
        start = end
        slot +=1
    
    return u_map, labels

def _map_to_img(prediction, shape, colours, metric=None, th=0.0, bg=(0, 0, 0)):
    """Generates an RGB image from `prediction` and `colours`
    
    The prediction itself should represent the index of its
    correspondent color.
    
    Parameters
    ----------
    prediction : array_like
        Array with the values to represent.
    shape : tuple of ints
        Original shape to reconstruct the image (without channels, just
        height and width).
    colours : list of RGB tuples
        List of colours for the RGB image representation.
    metric : array_like, optional (Default: None)
        Array with the same length of `prediction` to determine a
        metric for plotting or not each `prediction` value according to
        a threshold.
    th : float, optional (Default: 0.0)
        Threshold value to compare with each `metric` value if defined.
    bg : RGB tuple, optional (Default: (0, 0, 0))
        Background color used for the pixels not represented according
        to `metric`.
    
    Returns
    -------
    img : ndarray
        RGB image representation of `prediction` colouring each group
        according to `colours`.
    """
    
    # Generate RGB image shape
    img_shape = (shape[0], shape[1], 3)
    
    if metric is not None:
        
        # Coloured RGB image that only shows those values where metric
        # is lower to threshold
        return np.reshape([colours[int(p)] if m < th else bg
                           for p, m in zip(prediction, metric)], img_shape)
    else:
        
        # Coloured RGB image of the entire prediction
        return np.reshape([colours[int(p)] for p in prediction], img_shape)

# PLOT FUNCTIONS
# =============================================================================

def plot_reliability_diagram(output_dir, data, w, h, colours, num_groups=10):
    """Generates and saves the `reliability diagram` plot
    
    It saves the plot in `output_dir` in pdf format with the name
    `reliability_diagram.pdf`.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    data : dict
        It contains the `reliability diagram` data of each dataset. The
        key must be the dataset name abbreviation.
    w : int
        Width of the plot.
    h : int
        Height of the plot.
    colours : dict
        It contains the HEX value of the RGB color of each dataset. The
        key must be the dataset name abbreviation.
    num_groups : int, optional (default: 10)
        Number of groups to divide xticks labels.
    """
    
    # Generate x axis labels and data for the optimal calibration curve
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
    for img_name in colours.keys():
        ax.plot(xticks[:len(data[img_name])], data[img_name], label=img_name,
                color=colours[img_name])
    ax.plot(xticks, optimal, label="Optimal calibration", color='black',
            linestyle='dashed')
    
    # Axes labels
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
    file_name = "reliability_diagram.pdf"
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print("Saved {} in {}".format(file_name, output_dir), flush=True)

def plot_accuracy_vs_uncertainty(output_dir, acc_data, px_data, w, h, colours,
                                 H_limit=1.5, num_groups=15):
    """Generates and saves the `accuracy vs uncertainty` plot
    
    It saves the plot in `output_dir` in pdf format with the name
    `accuracy_vs_uncertainty.pdf`.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    acc_data : dict
        It contains the `accuracy vs uncertainty` data of each dataset.
        The key must be the dataset name abbreviation.
    px_data : dict
        It contains, for each dataset, the percentage of pixels of each
        uncertainty group. The key must be the dataset name
        abbreviation.
    w : int
        Width of the plot.
    h : int
        Height of the plot.
    colours : dict
        It contains the HEX value of the RGB color of each dataset. The
        key must be the dataset name abbreviation.
    H_limit : float, optional (default: 1.5)
        The max value of the range of uncertainty for the plot.
    num_groups : int, optional (default: 15)
        Number of groups to divide xticks labels.
    """
    
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
    for img_name in colours.keys():
        ax.plot(xticks[:len(acc_data[img_name])], acc_data[img_name],
                label="{} acc.".format(img_name), color=colours[img_name],
                zorder=3)
        ax.bar(xticks[:len(px_data[img_name])], px_data[img_name],
               label="{} px %".format(img_name), color=colours[img_name],
               alpha=0.18, zorder=2)
        ax.bar(xticks[:len(px_data[img_name])],
               [-0.007 for i in px_data[img_name]], bottom=px_data[img_name],
               color=colours[img_name], zorder=3)
    
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
    
    # Get legend handles and labels
    lg_handles, lg_labels = plt.gca().get_legend_handles_labels()
    
    # Manual legend to adjust the handles and place labels in a new order
    order = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
    ax.add_artist(ax.legend([lg_handles[idx] for idx in order],
                            [lg_labels[idx] for idx in order],
                            loc='upper center', ncol=5,
                            bbox_to_anchor=(0.46, 1.2)))
    
    # Manually added handles upper lines (to match the bars)
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
    file_name = "accuracy_vs_uncertainty.pdf"
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print("Saved {} in {}".format(file_name, output_dir), flush=True)

def plot_class_uncertainty(output_dir, name, epoch, avg_Ep, avg_H_Ep, w, h,
                           colours):
    """Generates and saves the `class uncertainty` plot of a dataset
    
    It saves the plot in `output_dir` in pdf format with the name
    `<NAME>_<EPOCH>_class_uncertainty.pdf`, where <NAME> is the
    abbreviation of the dataset name and <EPOCH> the number of trained
    epochs of the tested checkpoint.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    name : str
        The abbreviation of the dataset name.
    epoch : int
        The number of trained epochs of the tested checkpoint.
    avg_Ep : ndarray
        List of the averages of the aleatoric uncertainty (Ep) of each
        class. The last position also contains the average of the
        entire image.
    avg_H_Ep : ndarray
        List of the averages of the epistemic uncertainty (H - Ep) of
        each class. The last position also contains the average of the
        entire image.
    w : int
        Width of the plot.
    h : int
        Height of the plot.
    colours : dict
        It contains the HEX value of the RGB color of each dataset. The
        key must be the dataset name abbreviation.
    """
    
    # Xticks
    xticks = np.arange(len(avg_Ep))
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
    ax.bar(xticks, avg_Ep, label="Ep", color=colours["BO"], zorder=3)
    ax.bar(xticks, avg_H_Ep, bottom=avg_Ep, label="H - Ep",
           color=colours["KSC"], zorder=3)
    
    # Highlight avg border
    ax.bar(xticks[-1], avg_Ep[-1] + avg_H_Ep[-1], zorder=2,
           edgecolor=colours["IP"], linewidth=4)
    
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
    file_name = "{}_{}_class_uncertainty.pdf".format(name, epoch)
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print("Saved {} in {}".format(file_name, output_dir), flush=True)

def plot_maps(output_dir, name, shape, num_classes, wl, img, y, pred_map,
              H_map, colours, gradients, max_H=1.5, slots=15):
    """Generates and saves the `uncertainty map` plot of a dataset
    
    This plot shows an RGB representation of the hyperspectral image,
    the ground truth, the prediction map and the uncertainty map.
    
    It saves the plot in `output_dir` in pdf format with the name
    `H_<NAME>.pdf`, where <NAME> is the abbreviation of the dataset
    name.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    name : str
        The abbreviation of the dataset name.
    shape : tuple of ints
        Original shape to reconstruct the image (without channels, just
        height and width).
    num_classes : int
        Number of classes of the dataset.
    wl : list of floats
        Selected wavelengths of the hyperspectral image for RGB
        representation.
    img : ndarray
        Flattened list of the hyperspectral image pixels normalised.
    y : ndarray
        Flattened ground truth pixels of the hyperspectral image.
    pred_map : ndarray
        Array with the averages of the bayesian predictions.
    H_map : ndarray
        Array with the global uncertainty (H) values.
    colours : list of RGB tuples
        List of colours for the prediction map classes.
    gradients : list of RGB tuples
        List of colours for the uncertainty map groups of values.
    max_H : float, optional (default: 1.5)
        The max value of the range of uncertainty for the uncertainty
        map.
    slots : int, optional (default: 15)
        Number of groups to divide uncertainty map values.
    """
    
    # PREPARE FIGURE
    # -------------------------------------------------------------------------
    
    # Select shape and size depending on the dataset
    if name in ["IP", "KSC"]:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.set_size_inches(2*shape[1]/96, 2*shape[0]/96)
    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        fig.set_size_inches(4*shape[1]/96, shape[0]/96)
    
    # Remove axis
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    ax4.set_axis_off()
    
    # RGB IMAGE GENERATION
    #     Using HSI2RGB algorithm from paper:
    #         M. Magnusson, J. Sigurdsson, S. E. Armansson, M. O.
    #         Ulfarsson, H. Deborah and J. R. Sveinsson, "Creating RGB
    #         Images from Hyperspectral Images Using a Color Matching
    #         Function," IGARSS 2020 - 2020 IEEE International
    #         Geoscience and Remote Sensing Symposium, 2020,
    #         pp. 2045-2048, doi: 10.1109/IGARSS39084.2020.9323397.
    #     HSI2RGB code from:
    #         https://github.com/JakobSig/HSI2RGB
    # -------------------------------------------------------------------------
    
    # Create and show RGB image (D65 illuminant and 0.002 threshold)
    RGB_img = HSI2RGB(wl, img, shape[0], shape[1], 65, 0.002)
    ax1.imshow(RGB_img)
    
    # GROUND TRUTH GENERATION
    # -------------------------------------------------------------------------
    
    # Generate and show coloured ground truth
    gt = _map_to_img(y, shape, [(0, 0, 0)] + colours[:num_classes])
    ax2.imshow(gt)
    
    # PREDICTION MAP GENERATION
    # -------------------------------------------------------------------------
    
    # Generate and show coloured prediction map
    pred_H_img = _map_to_img(pred_map, shape, colours[:num_classes])
    ax3.imshow(pred_H_img)
    
    # UNCERTAINTY MAP GENERATION
    # -------------------------------------------------------------------------
    
    # Create uncertainty map
    u_map, labels = _uncertainty_to_map(H_map, num_classes, slots=slots,
                                        max_H=max_H)
    
    # Generate and show coloured uncertainty map
    H_img = _map_to_img(u_map, shape, gradients[:slots])
    ax4.imshow(H_img)
    
    # PLOT COMBINED IMAGE
    # -------------------------------------------------------------------------
    
    # Adjust layout between images
    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    
    # Save
    file_name = "H_{}.pdf".format(name)
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print("Saved {} in {}".format(file_name, output_dir), flush=True)

def plot_uncertainty_with_noise(output_dir, name, labels, data, w, h, colours):
    """Generates and saves the `noise` plot of a dataset
    
    It saves the plot in `output_dir` in pdf format with the name
    `<NAME>_noise.pdf`, where <NAME> is the abbreviation of the dataset
    name.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    name : str
        The abbreviation of the dataset name.
    labels : ndarray
        List of the evaluated noises that will be used as xlabels.
    data : ndarray
        It contains the class predictions for the list of noises of
        the dataset. The two last positions correspond to the average
        and the maximum uncertainty value prepared to be plotted.
    w : int
        Width of the plot.
    h : int
        Height of the plot.
    colours : dict
        It contains the HEX value of the RGB color of each dataset. The
        key must be the dataset name abbreviation.
    """
    
    # Add the data for plotting the maximum uncertainty line
    max_uncertainty = np.log(len(data) - 1)
    data = np.concatenate((data, [[max_uncertainty] * len(data[0])]))
    
    # Labels and xticks
    xticks = np.linspace(0.0, labels[-1], 13)
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
    #     Some of the colours of other plots are used here o it is no
    #     necessary to define different colours for this plot
    for n, d in enumerate(data[:-3]):
        ax.plot(labels, d, color=colours["BO"])
    ax.plot(labels, data[-3], color=colours["BO"], label="classes")
    ax.plot(labels, data[-2], color=colours["IP"], label="avg",
            linestyle='dashed')
    ax.plot(labels, data[-1], color=colours["KSC"], label="max",
            linestyle='dashed')
    
    # Axes label
    ax.set_xlabel("Noise factor")
    ax.set_ylabel("Uncertainty")
    
    # Y axis limit
    y_lim = np.ceil(max_uncertainty)
    if max_uncertainty <= y_lim - 0.5:
        y_lim -= 0.5
    ax.set_ylim((0, y_lim))
    
    # X axis limit
    ax.set_xlim((xticks[0], xticks[-1]))
    
    # X axis labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.around(xticks, 2))
    
    # Grid
    ax.grid(axis='y')
    
    # Legend
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
    
    # Save
    file_name = "{}_noise.pdf".format(name)
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print("Saved {} in {}".format(file_name, output_dir), flush=True)

def plot_combined_noise(output_dir, labels, data, w, h, colours):
    """Generates and saves the `combined noise` plot
    
    It saves the plot in `output_dir` in pdf format with the name
    `combined_noise.pdf`.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    labels : ndarray
        List of the evaluated noises that will be used as xlabels.
    data : dict
        It contains the normalised average predictions for the list of
        noises of each dataset. The key must be the dataset name
        abbreviation.
    w : int
        Width of the plot.
    h : int
        Height of the plot.
    colours : dict
        It contains the HEX value of the RGB color of each dataset. The
        key must be the dataset name abbreviation.
    """
    
    # Labels and xticks
    xticks = np.linspace(0.0, labels[-1], 11)
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
    for name, d in data.items():
        ax.plot(labels, d, color=colours[name], label=name)
    
    # Axes label
    ax.set_xlabel("Noise factor")
    ax.set_ylabel("Uncertainty")
    
    # Y axis limit
    ax.set_ylim((0, 1))
    
    # X axis limit
    ax.set_xlim((xticks[0], xticks[-1]))
    
    # X axis labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.around(xticks, 2))
    
    # Grid
    ax.grid(axis='y')
    
    # Legend
    ax.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.2))
    
    # Save
    file_name = "combined_noise.pdf"
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print("Saved {} in {}".format(file_name, output_dir), flush=True)

def plot_mixed_uncertainty(output_dir, name, epoch, data, class_a, class_b, w,
                           h, colours):
    """Generates and saves the `mixed classes` plot of a dataset
    
    It saves the plot in `output_dir` in pdf format with the name
    `<NAME>_<EPOCH>_mixed_classes.pdf`, where <NAME> is the
    abbreviation of the dataset name and <EPOCH> the number of trained
    epochs of the tested checkpoint.
    
    Parameters
    ----------
    output_dir : str
        Path of the output directory. It can be an absolute path or
        relative from the execution path.
    name : str
        The abbreviation of the dataset name.
    epoch : int
        The number of trained epochs of the tested checkpoint.
    data : list of lists
        Contains one list for the normal model predictions with the
        aleatoric uncertainty (Ep) values of the mixed classes and the
        dataset average, and other list with the same information for
        the mixed model predictions.
    class_a : int
        Number of the first mixed class.
    class_b : int
        Number of the second mixed class.
    w : int
        Width of the plot.
    h : int
        Height of the plot.
    colours : dict
        It contains the HEX value of the RGB color of each dataset. The
        key must be the dataset name abbreviation.
    """
    
    # Xticks
    xticks = np.arange(len(data[0]))
    xticks_0 = xticks - 0.21
    xticks_1 = xticks + 0.21
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
    ax.bar(xticks_0, data[0], label="Ep", color=colours["BO"], width=0.35,
           zorder=3)
    ax.bar(xticks_1, data[1], label="Ep mixed", color=colours["SV"],
           width=0.35, zorder=3)
    
    # Axes label
    ax.set_xlabel("{} mixed classes".format(name))
    
    # X axis labels
    ax.set_xticks(xticks)
    xlabels = ["class {}".format(class_a), "class {}".format(class_b),
               "avg. (all classes)"]
    ax.set_xticklabels(xlabels)
    
    # Grid
    ax.grid(axis='y', zorder=1)
    
    # Legend
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
    
    # Save
    file_name = "{}_{}_mixed_classes.pdf".format(name, epoch)
    plt.savefig(os.path.join(output_dir, file_name), bbox_inches='tight')
    print("Saved {} in {}".format(file_name, output_dir), flush=True)

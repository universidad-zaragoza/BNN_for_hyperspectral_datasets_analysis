#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot module of the BNN4HI package

The functions of this module are used to generate plots using the
results of the analised bayesian predictions.
"""

__version__ = "1.0.0"
__author__ = "Adrián Alcolea"
__email__ = "alcolea@unizar.es"
__maintainer__ = "Adrián Alcolea"
__license__ = "GPLv3"
__credits__ = ["Adrián Alcolea", "Javier Resano"]

import os
import numpy as np
import matplotlib.pyplot as plt

# PLOT FUNCTIONS
# ==============================================================================

def plot_reliability_diagram(output_dir, data, w, h, colours, num_groups=10):
    
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
    for img_name in colours.keys():
        ax.plot(xticks[:len(data[img_name])], data[img_name], label=img_name,
                color=colours[img_name])
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

def plot_accuracy_vs_uncertainty(output_dir, acc_data, px_data, w, h, colours,
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

def plot_class_uncertainty(output_dir, name, epoch, avg_Ep, avg_H_Ep, w, h,
                           colours):
    
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

def plot_uncertainty_with_noise(output_dir, name, labels, data, w, h, colours):
    
    # Add the data for plotting the maximum uncertainty line
    max_uncertainty = np.log(len(data) - 1)
    data = np.concatenate((data, [[max_uncertainty] * len(data[0])]))
    
    # Labels and xticks
    xticks = np.linspace(0.0, labels[-1], 13)
    
    # Generate figure and axis
    fig, ax = plt.subplots(figsize=(w, h))
    
    # Plots
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
#    ax.set_xticklabels(xticks)
    ax.set_xticklabels(np.around(xticks, 2))
    
    # Grid
    ax.grid(axis='y')
    
    # Legend
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
    
    # Save
    plt.savefig(os.path.join(output_dir, "{}_noise.pdf".format(name)),
                bbox_inches='tight')

def plot_combined_noise(output_dir, labels, data, w, h, colours):
    
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
#    y_lim = np.ceil(max_uncertainty)
#    if max_uncertainty <= y_lim - 0.5:
#        y_lim -= 0.5
    ax.set_ylim((0, 1))
    
    # X axis limit
    ax.set_xlim((xticks[0], xticks[-1]))
    
    # X axis labels
    ax.set_xticks(xticks)
#    ax.set_xticklabels(xticks)
    ax.set_xticklabels(np.around(xticks, 2))
    
    # Grid
    ax.grid(axis='y')
    
    # Legend
    ax.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.2))
    
    # Save
    plt.savefig(os.path.join(output_dir, "combined_noise.pdf".format(name)),
                bbox_inches='tight')

def plot_mixed_uncertainty(output_dir, name, epoch, data, class_a, class_b, w,
                           h, colours):
    
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

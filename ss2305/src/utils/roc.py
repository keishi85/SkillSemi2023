"""
ROC analysis for 2-class classification
Created on Tue Nov 16 2021
@author: ynomura
"""

import numpy as np
import matplotlib.pyplot as plt


def roc_analysis(likelihoods, labels, roc_curve_file_name):

    # sort by likelihood (descending order)
    sorted_idx = np.argsort(-likelihoods)
    binary_array = labels[sorted_idx]
    sorted_likelihood = likelihoods[sorted_idx]

    # ROC analysis
    fpr = np.cumsum(1 - binary_array) / np.sum(1 - binary_array)
    tpr = np.cumsum(binary_array) / np.sum(binary_array)
    auc = np.sum(tpr * (1 - binary_array)) / np.sum(1 - binary_array)

    # Add (0,0) for plotting
    fpr_for_plot = np.insert(fpr, 0, 0.0)
    tpr_for_plot = np.insert(tpr, 0, 0.0)

    # Get cut-off point by Youden index
    cutoff_idx = np.argmax(tpr - fpr)
    max_accuracy = (np.sum(binary_array[0:cutoff_idx + 1] == 1)
                    + np.sum(binary_array[cutoff_idx + 1:] == 0)) / len(tpr)

    # Plot ROC curve
    plt.plot(fpr_for_plot, tpr_for_plot, 'b', linewidth=2.0, clip_on=False)

    base_points = np.array([0, 1])
    plt.plot(base_points, base_points, 'k', linestyle='dotted')
    plt.plot(fpr[cutoff_idx], tpr[cutoff_idx], 'o', ms=10, markeredgewidth=2,
             markerfacecolor="None", markeredgecolor='k')

    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (TPR)')
        
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.text(0.5, 0.3, 'AUC=%.3f' % (auc), size=15)

    plt.savefig(roc_curve_file_name)
    plt.close()

    return auc, tpr[cutoff_idx], fpr[cutoff_idx], sorted_likelihood[cutoff_idx]


if __name__ == '__main__':

    likelihoods = np.array([0.210, 0.327, 0.451, 0.930, 0.649,
                            0.772, 0.481, 0.601, 0.308, 0.852,
                            0.758, 0.379, 0.268, 0.368, 0.770,
                            0.250, 0.396, 0.121, 0.637, 0.523,
                            0.204, 0.587, 0.894, 0.078, 0.773,
                            0.083, 0.639, 0.773, 0.421, 0.820,
                            0.101, 0.348, 0.393, 0.309, 0.598,
                            0.392, 0.532, 0.398, 0.285, 0.352,
                            0.610, 0.641, 0.582, 0.447, 0.940,
                            0.691, 0.353, 0.377, 0.388, 0.154])

    labels = np.array([1, 0, 0, 1, 0,
                       1, 0, 1, 0, 1,
                       1, 1, 0, 0, 1,
                       0, 1, 0, 1, 0,
                       0, 1, 0, 0, 0,
                       0, 1, 1, 0, 1,
                       1, 0, 1, 0, 1,
                       0, 1, 0, 1, 0,
                       1, 1, 1, 0, 1,
                       1, 0, 0, 1, 0])

    roc_curve_file_name = "roc_curve.png"

    roc_analysis(likelihoods, labels, roc_curve_file_name)

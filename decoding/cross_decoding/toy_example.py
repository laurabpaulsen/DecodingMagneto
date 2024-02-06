"""
This script investigates the correlation between the decoding accuracies and the distance between the sessions.
- The distances are defined both as days and as the number of sessions.
- The accuracies are calculated for the visual and the memory sessions separately, as well as for the combined sessions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from pandas import to_datetime

from dist_acc_corr import get_distance_matrix, prep_x_y

# set parameters for all plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['image.cmap'] = 'RdBu_r'
plt.rcParams['image.interpolation'] = 'bilinear'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300



def plot_corr_toy_example(acc, save_path = None, corr_color="C0", perm_color="lightblue", alpha=0.05, timepoint=0):
    # only visual

    # convert to datetime
    dates = to_datetime(['08-10-2020', '09-10-2020', '15-10-2020'], format='%d-%m-%Y')
       
    # set up figure
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)

    dist = get_distance_matrix(dates)

    # get x and y
    X, y = prep_x_y(acc, dist)

    annotations = [f"{i+1, j+1}" for i in range(3) for j in range(3) if i != j]

    # plot correlation for the first time point
    ax.scatter(y, X[:, timepoint], color="k")
    ax.set_xlabel("Distance")
    ax.set_ylabel(f"Accuracy (t={timepoint+1})")

    # annotate each point with the session number of both training and testing
    for i, txt in enumerate(annotations):
        # get the location of the annotation
        x_coord = y[i] - 0.5
        y_coord = X[i, timepoint] - 1/150

        # annotate
        ax.annotate(txt, (x_coord, y_coord))
    
    ax.set_xlim([0, 8])
    ax.set_ylim([0.45, 0.55])

    
            
                
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

def plot_corr_toy_example_cond(acc, save_path = None, corr_color="C0", perm_color="lightblue", alpha=0.05, timepoint=0):

       
    # set up figure
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)

    dist = get_distance_matrix([0, 1, 1])

    # get x and y
    X, y = prep_x_y(acc, dist)

    annotations = [f"{i+1, j+1}" for i in range(3) for j in range(3) if i != j]

    # plot correlation for the first time point
    ax.scatter(y, X[:, timepoint], color="k")
    ax.set_xlabel("Distance")
    ax.set_ylabel(f"Accuracy (t={timepoint+1})")

    # annotate each point with the session number of both training and testing
    for i, txt in enumerate(annotations):
        # get the location of the annotation
        x_coord = y[i] - 0.5
        y_coord = X[i, timepoint] - 1/150

        # annotate
        ax.annotate(txt, (x_coord, y_coord))
    
    ax.set_xlim([-1, 2])
    ax.set_ylim([0.45, 0.55])


                
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

if __name__ == "__main__":
    path = Path(__file__)

    # load accuracies from cross decoding
    acc = np.load(path.parents[0] / "accuracies" / f"cross_decoding_10_LDA_sens.npy", allow_pickle=True)

    # output path
    plot_path = path.parents[0] / "plots" 

    # only include the first three sessions
    acc = acc[:3, :, :, :][:, :3, :, :]

    # plot correlation for toy example
    plot_corr_toy_example(acc, save_path = plot_path / "corr_toy_example.png")

    # plot correlation for toy example with condition
    plot_corr_toy_example_cond(acc, save_path = plot_path / "corr_toy_example_cond.png")
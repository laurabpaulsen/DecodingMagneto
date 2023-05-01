"""
This script investigates the correlation between the decoding accuracies and the distance between the sessions.
- The distances are defined both as days and as the number of sessions.
- The accuracies are calculated for the visual and the memory sessions separately, as well as for the combined sessions.
"""

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# get pearsons r
from scipy.stats import pearsonr

# dates from pandas
from pandas import to_datetime

# set parameters for all plots
plt.rcParams['font.family'] = 'times new roman'
plt.rcParams['image.cmap'] = 'RdBu_r'
plt.rcParams['image.interpolation'] = 'bilinear'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300


def x_axis_seconds(ax):
    """
    Changes the x axis to seconds
    """
    ax.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])

def get_distance_matrix(order):
    # number of sessions
    n_sessions = len(order)

    # initialize distance matrix
    dist = np.zeros((n_sessions, n_sessions))

    # fill distance matrix
    for i in range(n_sessions):
        for j in range(n_sessions):
            try: # if order is a list of integers
                dist[i, j] = abs(order[i] - order[j])
            except: # if order is a list of datetime objects
                dist[i, j] = abs((order[i] - order[j]).days)

    return dist


def prep_x_y(acc, dist):
    # initialize lists
    X = []
    y = []

    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            if dist[i,j] != 0: # do not include within session decoding
                X.append(np.diag(acc[i, j, :, :]))
                y.append(dist[i, j])
    
    return X, y

def get_corr_pval(X, y):
    # initialize lists
    corr = []
    pval = []

    for t in range(X[0].shape[0]):
        x = [x[t] for x in X]
        r, p = pearsonr(x, y)
        corr.append(r)
        pval.append(p)

    return np.array(corr), np.array(pval)

def plot_corr(ax, corr, pval):
    # plot correlation
    ax.plot(corr, color="darkblue", linewidth=1.5)

    # plot significance
    sig = np.where(pval < 0.05)[0]
    min_corr = np.min(corr)
    bottom = np.full(sig.shape, min_corr - 0.1)

    ax.plot(sig, bottom, "*", color="black", markersize=3, alpha = 0.8)

    x_axis_seconds(ax)

    # set limits
    ax.set_xlim([0, 250])



def main():
    path = Path(__file__).parent

    # load accuracies
    acc = np.load(path / "accuracies" / f"cross_decoding_10_LDA_sens.npy", allow_pickle=True)

    # only memory
    mem_indices = [4, 5, 6, 7]
    mem_acc = acc[mem_indices, :, :, :][:, mem_indices, :, :]

    # only visual
    vis_indices = [0, 1, 2, 3, 8, 9, 10]
    vis_acc = acc[vis_indices, :, :, :][:, vis_indices, :, :]

    # order of sessions
    order = [0, 1, 2, 3, 7, 8, 9, 10, 4, 5, 6] 

    # convert to datetime
    dates = to_datetime(['08-10-2020', '09-10-2020', '15-10-2020', '16-10-2020', '02-03-2021', '16-03-2021', '18-03-2021', '22-10-2020', '29-10-2020', '12-11-2020', '13-11-2020'], format='%d-%m-%Y')

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), dpi=300)

    for i, (acc, inds) in enumerate(zip([vis_acc, mem_acc, acc], [vis_indices, mem_indices, None])):
        for j, dist_type in enumerate(["dates", "order"]):
            if dist_type == "dates":
                dist = get_distance_matrix(dates)
            elif dist_type == "order":
                dist = get_distance_matrix(order)

            # use indices for visual and memory, not for combined
            if inds is not None:
                dist = dist[inds, :][:, inds]

            # get x and y
            X, y = prep_x_y(acc, dist)

            # get correlation and p-values
            corr, pval = get_corr_pval(X, y)

            # plot
            plot_corr(axes[i, j], corr, pval)

    fig.supxlabel("Time (s)", fontsize=16)
    fig.supylabel("Pearson's r", fontsize=16)
    fig.suptitle("Correlation between decoding accuracy and distance between sessions", fontsize=20)
    
    # first column y label
    axes[0, 0].set_ylabel("Visual")
    axes[1, 0].set_ylabel("Memory")
    axes[2, 0].set_ylabel("Combined")

    # first row x label
    axes[0, 0].set_title("Days")
    axes[0, 1].set_title("Sessions")

    # save huge figure
    plt.tight_layout()
    plt.savefig(path / "plots" / f"corr_acc_dist.png")


        


if __name__ == "__main__":
    main()
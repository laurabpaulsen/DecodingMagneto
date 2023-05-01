"""
This script investigates the correlation between the decoding accuracies and the distance between the bins.
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
            if dist[i,j] != 0: # do not include within bin decoding
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
    acc = np.load(path / "accuracies" / f"LDA_auto_10.npy", allow_pickle=True)

    # order of sessions
    order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 

    fig, ax = plt.subplots(1, figsize=(7, 5), dpi=300)

    dist = get_distance_matrix(order)

    # get x and y
    X, y = prep_x_y(acc, dist)

    # get correlation and p-values
    corr, pval = get_corr_pval(X, y)

    # plot
    plot_corr(ax, corr, pval)

    fig.supxlabel("Time (s)", fontsize=14)
    fig.supylabel("Pearson's r", fontsize=14)
    fig.suptitle("Correlation between decoding accuracy and distance between bins", fontsize=16)


    # save huge figure
    plt.tight_layout()
    plt.savefig(path / "plots" / f"corr_acc_dist.png")


if __name__ == "__main__":
    main()
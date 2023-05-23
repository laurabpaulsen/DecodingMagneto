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

# t-test
from scipy.stats import ttest_1samp

# set parameters for all plots
plt.rcParams['font.family'] = 'Serif'
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


def plot_correlations(acc, save_path = None):
    
    # only memory
    mem_indices = [4, 5, 6, 7]
    mem_acc = acc[mem_indices, :, :, :][:, mem_indices, :, :]

    # only visual
    vis_indices = [0, 1, 2, 3, 8, 9, 10]
    vis_acc = acc[vis_indices, :, :, :][:, vis_indices, :, :]

    # order of sessions
    order = [0, 1, 2, 3, 7, 8, 9, 10, 4, 5, 6] 

    bin_range = (-0.75, 0.75)
    bins = np.linspace(bin_range[0], bin_range[1], 40)

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), dpi=300)

    for i, (acc, inds) in enumerate(zip([vis_acc, mem_acc, acc], [vis_indices, mem_indices, None])):
        dist = get_distance_matrix(order)

        # use indices for visual and memory, not for combined
        if inds is not None:
            dist = dist[inds, :][:, inds]

        # get x and y
        X, y = prep_x_y(acc, dist)

        # get correlation and p-values
        corr, pval = get_corr_pval(X, y)

        # plot
        plot_corr(axes[i, 0], corr, pval)

        t, p = ttest_1samp(corr, 0)
        print(f"Mean correlation: {np.mean(corr):.3f}, p-value: {p:.3f}")


        # plot histogram of correlations
        axes[i, 1].hist(corr, bins = bins, color="lightblue")

        # add mean correlation and p-value
        axes[i, 1].text(0.05, 0.95, f"Mean: {np.mean(corr):.3f}\np-value: {p:.3f}", transform=axes[i, 1].transAxes, verticalalignment='top', horizontalalignment='left', fontsize=10)

        # vertical line at mean
        axes[i, 1].axvline(np.mean(corr), color="k", linewidth=1, linestyle="--", label="Mean")

    fig.supxlabel("Time (s)", fontsize=16)
    fig.supylabel("Pearson's r", fontsize=16)
    fig.suptitle("Bins", fontsize=20)
    
    # first column y label
    axes[0, 0].set_ylabel("Visual")
    axes[1, 0].set_ylabel("Memory")
    axes[2, 0].set_ylabel("Combined")

    # first row x label
    axes[0, 0].set_title("Correlation")
    axes[0, 1].set_title("Histogram of correlations")


    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path )


def main():
    path = Path(__file__)

    # load accuracies
    acc = np.load(path.parents[0] / "accuracies" / f"LDA_auto_10.npy", allow_pickle=True)
    plot_path = path.parents[0] / "plots" 

    # plot
    plot_correlations(acc, save_path=plot_path / "corr_acc_dist.png")

        

if __name__ == "__main__":
    main()
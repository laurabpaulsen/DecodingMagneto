"""

"""

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# get pearsons r
from scipy.stats import pearsonr

# t-test
from scipy.stats import ttest_1samp

# set parameters for all plots
plt.rcParams['font.family'] = 'sans-serif'
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


def plot_corr_hist(vis_acc, mem_acc, all_acc, save_path = None):
    # set up figure
    gs_kw = dict(width_ratios=[1, 0.4], height_ratios=[1, 1, 1], wspace=0.01, hspace=0.3)
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), dpi=300, gridspec_kw=gs_kw, sharey=True)

    bin_range = (-0.4, 0.4)

    bins = np.linspace(bin_range[0], bin_range[1], 40)

    for i, acc in enumerate([vis_acc, mem_acc, all_acc]):
        dist = get_distance_matrix(range(acc.shape[0]))
        ax_hist = axes[i, 1]
        ax_corr = axes[i, 0]


        # get x and y
        X, y = prep_x_y(acc, dist)

        # get correlation and p-values
        corr, pval = get_corr_pval(X, y)

        # test if mean of correlations is significantly different from 0
        t, p = ttest_1samp(corr, 0)
        print(f"Mean correlation: {np.mean(corr):.3f}, p-value: {p:.3f}")

        # plot correlation
        ax_corr.plot(corr) 
            
        # seconds on x axis
        x_axis_seconds(ax_corr)
            
        # set limits
        ax_corr.set_xlim([0, 250])
        
        # plot histogram of correlations
        ax_hist.hist(corr, bins = bins, color="lightblue", orientation="horizontal")
        ax_hist.set_axis_off()

        # vertical line at mean
        ax_hist.axhline(np.mean(corr), color="k", linewidth=1, linestyle="--", label="Mean")

        # add mean correlation and p-value
        ax_hist.text(0.05, 0.95, f"Mean: {np.mean(corr):.3f}\np-value: {p:.3f}", transform=ax_hist.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=10)
            

    axes[2, 0].set_xlabel("TIME (s)", fontsize=16)
    fig.supylabel("PEARSON'S R", fontsize=16)

    
    axes[0, 0].set_ylabel("Visual".upper())
    axes[1, 0].set_ylabel("Memory".upper())
    axes[2, 0].set_ylabel("Combined".upper())
    

    # share title between columns
    axes[0, 0].set_title("bins".upper())

    if save_path is not None:
        plt.savefig(save_path )

def main():
    path = Path(__file__)

    # load accuracies
    acc_all = np.load(path.parents[0] / "accuracies" / f"LDA_auto_10_all_11.npy", allow_pickle=True)
    acc_vis = np.load(path.parents[0] / "accuracies" / f"LDA_auto_10_visual_11.npy", allow_pickle=True)
    acc_mem = np.load(path.parents[0] / "accuracies" / f"LDA_auto_10_memory_11.npy", allow_pickle=True)
    
    plot_path = path.parents[0] / "plots" 

    # plot
    plot_corr_hist(acc_vis, acc_mem, acc_all, save_path=plot_path / "corr_acc_dist_bins.png")

        

if __name__ == "__main__":
    main()
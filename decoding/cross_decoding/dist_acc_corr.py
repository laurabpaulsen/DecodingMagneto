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
            if i != j: # do not include within session decoding
                X.append(np.diag(acc[i, j, :, :]))
                y.append(dist[i, j])
    
    return np.array(X), np.array(y)

def plot_corr(ax, corr, pval, alpha = 0.05, color = "lightblue", y_lim=None):
    # plot correlation
    ax.plot(corr, color=color, linewidth=1.5)

    if y_lim:
        ax.set_ylim(y_lim)

    # plot significance
    sig = np.where(pval < alpha)[0]
    min_corr = np.min(corr)
    bottom = np.full(sig.shape, y_lim[0] +  0.1 if y_lim else min_corr - 0.1)

    ax.plot(sig, bottom, "o", color="black", markersize=3, alpha = 0.8)

    x_axis_seconds(ax)

    # set limits
    ax.set_xlim([0, 250])



def plot_hist_of_corr(ax, corr, bins, color="lightblue", y_lim=(-0.5, 0.5)):
    ax.hist(corr, bins = bins, color=color, orientation="horizontal")
    ax.set_axis_off()

    # vertical line at mean
    ax.axhline(np.mean(corr), color="k", linewidth=1, linestyle="--", label="Mean")

    # limits
    ax.set_ylim(y_lim)


def permutation_test(X, y, n_perm):
    """
    Permutation test to see if correlation is significant using cluster permutation test

    Parameters
    ----------
    X : numpy array
        Array of shape (n_sessionpairs, n_timepoints) containing the decoding accuracies
    """
    # array for storing actual correlation
    corrs = np.zeros(X.shape[1])

    # initialize numpy array to store all permutations
    permutations_corr = np.zeros((n_perm, len(X[0])))

    # loop over time points
    for t in range(X.shape[1]):
        # actual correlation
        corrs[t], _ = pearsonr(X[:, t], y)

        # loop over all permutations
        for n in range(n_perm):
            # permute y
            perm_y = np.random.permutation(y)

            # calculate correlation
            perm_corr, _ = pearsonr(X[:, t], perm_y)

            # store permutation correlation
            permutations_corr[n, t] = perm_corr

    pvals = np.zeros(X.shape[1])
    
    # calculate p-value
    for t in range(X.shape[1]):
        pvals[t] = np.sum(abs(permutations_corr[:, t]) > abs(corrs[t])) / n_perm
        
    return corrs, permutations_corr, pvals

def plot_corr_hist_cond(acc, save_path = None, corr_color="C0", perm_color="lightblue", alpha=0.05):

    # prep x and y
    dist = get_distance_matrix([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    # get x and y
    X, y = prep_x_y(acc, dist)

    # permutation test to see if correlation is significant
    n_perm = 1000
    corr, all_perm, pvals = permutation_test(X, y, n_perm)

    # set up figure
    gs_kw = dict(width_ratios=[1, 0.4], height_ratios=[1], wspace=0.01, hspace=0.3)
    fig, ax = plt.subplots(1, 2, figsize=(10, 6), dpi=300, gridspec_kw=gs_kw)

    bin_range = (-0.5, 0.5)

    bins = np.linspace(bin_range[0], bin_range[1], 20)

    # plot permutations
    for perm in all_perm:
        ax[0].plot(perm, color=perm_color, linewidth=0.5, alpha=0.4)

    plot_corr(ax[0], corr, pvals, alpha = alpha, color=corr_color)
    # get y limits
    y_lim = ax[0].get_ylim()

    # plot histogram of correlations
    plot_hist_of_corr(ax[1], corr, bins, color=corr_color, y_lim=None)


    # add y label
    ax[0].set_ylabel("PEARSON'S R", fontsize=16)

    # add x label
    ax[0].set_xlabel("TIME (s)", fontsize=16)


    if save_path is not None:
        plt.savefig(save_path)


def plot_corr_hist_no_combined(acc, save_path = None, corr_color="C0", perm_color="lightblue", alpha=0.05):
    # only memory
    mem_indices = [7, 8, 9, 10]
    mem_acc = acc[mem_indices, :, :, :][:, mem_indices, :, :]

    # only visual
    vis_indices = [0, 1, 2, 3, 4, 5, 6]
    vis_acc = acc[vis_indices, :, :, :][:, vis_indices, :, :]

    # convert to datetime
    dates = to_datetime(['08-10-2020', '09-10-2020', '15-10-2020', '16-10-2020', '02-03-2021', '16-03-2021', '18-03-2021', '22-10-2020', '29-10-2020', '12-11-2020', '13-11-2020'], format='%d-%m-%Y')
    order = np.array([0, 1, 2, 3, 8, 9, 10, 4, 5, 6, 7])
       
    # set up figure
    gs_kw = dict(width_ratios=[1, 0.4, 1, 0.4], height_ratios=[1, 1], wspace=0.01, hspace=0.3)
    fig, axes = plt.subplots(2, 4, figsize=(12, 8), dpi=300, gridspec_kw=gs_kw, sharey=True)

    bin_range = (-0.65, 0.65)

    bins = np.linspace(bin_range[0], bin_range[1], 20)

    for i, (tmp_acc, inds) in enumerate(zip([vis_acc, mem_acc], [vis_indices, mem_indices])):

        tmp_dates = dates.copy()[inds] if inds is not None else dates.copy()
        tmp_order = [i for i in order if i in inds] if inds is not None else order.copy()

        for j, dist_type in enumerate(["days", "session"]):
            if dist_type == "days":
                dist = get_distance_matrix(tmp_dates)
            elif dist_type == "session":
                dist = get_distance_matrix(tmp_order)

            ax_hist = axes[i, j*2+1]
            ax_corr = axes[i, j*2]

            # get x and y
            X, y = prep_x_y(tmp_acc, dist)

            # permutation test to see if correlation is significant
            n_perm = 1000
            corr, all_perm, pvals = permutation_test(X, y, n_perm)
            
            for perm in all_perm:
                ax_corr.plot(perm, color=perm_color, linewidth=0.5, alpha=0.4)

            # plot correlation
            plot_corr(ax_corr, corr, pvals, alpha = alpha, color=corr_color, y_lim=(-1, 1))

            # get y limits
            y_lim = ax_corr.get_ylim()

            # plot histogram of correlations
            plot_hist_of_corr(ax_hist, corr, bins, color=corr_color, y_lim=y_lim)


    fig.supxlabel("TIME (s)", fontsize=16)
    fig.supylabel("PEARSON'S R", fontsize=16)

    
    # first column y label
    axes[0, 0].set_ylabel("Visual".upper())
    axes[1, 0].set_ylabel("Memory".upper())

    # share title between columns
    axes[0, 0].set_title("Days".upper())
    axes[0, 2].set_title("Sessions".upper())

    if save_path is not None:
        plt.savefig(save_path)

if __name__ == "__main__":
    path = Path(__file__)

    # load accuracies from cross decoding
    acc = np.load(path.parents[0] / "accuracies" / f"cross_decoding_10_LDA_sens.npy", allow_pickle=True)

    # output path
    plot_path = path.parents[0] / "plots" 

    alpha = 0.001
    # plot
    #plot_corr_hist(acc = acc, save_path = plot_path / "corr_acc_dist.png")
    plot_corr_hist_no_combined(
        acc = acc, 
        save_path = plot_path / "corr_acc_dist_no_combined.png",
        alpha = alpha)

    plot_corr_hist_cond(
        acc, 
        save_path=plot_path / "corr_acc_dist_cond.png",
        alpha = alpha
        )
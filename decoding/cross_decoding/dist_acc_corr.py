"""
This script investigates the correlation between the decoding accuracies and the distance between the sessions.
- The distances are defined both as days and as the number of sessions.
- The accuracies are calculated for the visual and the memory sessions separately, as well as for the combined sessions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from pathlib import Path

# get pearsons r
from scipy.stats import pearsonr

# t-test
from scipy.stats import ttest_1samp

# dates from pandas
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
            if dist[i,j] != 0: # do not include within session decoding
                X.append(np.diag(acc[i, j, :, :]))
                y.append(dist[i, j])
    
    return X, y

def prep_x_y_cond(acc, dist):
    # initialize lists
    X = []
    y = []

    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            if i != j: # do not include within session decoding
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
    ax.plot(corr, color="lightblue", linewidth=1.5)

    # plot significance
    sig = np.where(pval < 0.05)[0]
    min_corr = np.min(corr)
    bottom = np.full(sig.shape, min_corr - 0.1)

    ax.plot(sig, bottom, "*", color="black", markersize=3, alpha = 0.8)

    x_axis_seconds(ax)

    # set limits
    ax.set_xlim([0, 250])

def plot_hist_of_corr(ax, corr, p, bins):
    ax.hist(corr, bins = bins, color="lightblue", orientation="horizontal")
    ax.set_axis_off()

    # vertical line at mean
    ax.axhline(np.mean(corr), color="k", linewidth=1, linestyle="--", label="Mean")

    # add mean correlation and p-value
    ax.text(0.05, 0.95, f"Mean: {np.mean(corr):.3f}\np-value: {p:.3f}", transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=14)


def plot_corr_hist(acc, save_path = None):
    # prepare dictinary for printing table
    table = {}

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
    gs_kw = dict(width_ratios=[1, 0.4, 1, 0.4], height_ratios=[1, 1, 1], wspace=0.01, hspace=0.3)
    fig, axes = plt.subplots(3, 4, figsize=(12, 8), dpi=300, gridspec_kw=gs_kw, sharey=True)

    bin_range = (-0.65, 0.65)

    bins = np.linspace(bin_range[0], bin_range[1], 20)

    for i, (tmp_acc, inds) in enumerate(zip([vis_acc, mem_acc, acc], [vis_indices, mem_indices, None])):

        condition = ["visual", "memory", "combined"][i]

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

            # get correlation and p-values
            corr, pval = get_corr_pval(X, y)

            # test if mean of correlations is significantly different from 0
            t, p = ttest_1samp(corr, 0)

            # add to table
            table[f'{i, j}'] = {
                "condition": condition,
                "distance type": dist_type,
                "mean correlation": np.mean(corr),
                "p-value": p}


            # plot correlation
            ax_corr.plot(corr) 
            # seconds on x axis
            x_axis_seconds(ax_corr)
            
            # set limits
            ax_corr.set_xlim([0, 250])

            # plot histogram of correlations
            plot_hist_of_corr(ax_hist, corr, p, bins)


    fig.supxlabel("TIME (s)", fontsize=16)
    fig.supylabel("PEARSON'S R", fontsize=16)

    
    # first column y label
    axes[0, 0].set_ylabel("Visual".upper())
    axes[1, 0].set_ylabel("Memory".upper())
    axes[2, 0].set_ylabel("Combined".upper())

    # share title between columns
    axes[0, 0].set_title("Days".upper())
    axes[0, 2].set_title("Sessions".upper())

    # add table
    table = pd.DataFrame.from_dict(table, orient="index")

    # round the values to 3 decimals
    table = table.round(3)

    # print table to console in latex format without index
    print(table.to_latex(index=False))


    if save_path is not None:
        plt.savefig(save_path)

def plot_corr_hist_cond(acc, save_path = None):

    # prep x and y
    dist = get_distance_matrix([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    # get x and y
    X, y = prep_x_y_cond(acc, dist)

    # get correlation and p-values
    corr, pval = get_corr_pval(X, y)

    # test if mean of correlations is significantly different from 0
    t, p = ttest_1samp(corr, 0)
    
    # set up figure
    gs_kw = dict(width_ratios=[1, 0.4], height_ratios=[1], wspace=0.01, hspace=0.3)
    fig, ax = plt.subplots(1, 2, figsize=(8, 6), dpi=300, gridspec_kw=gs_kw)

    bin_range = (-0.5, 0.5)

    bins = np.linspace(bin_range[0], bin_range[1], 20)

    # plot correlation
    ax[0].plot(corr)
    # seconds on x axis
    x_axis_seconds(ax[0])

    # set limits
    ax[0].set_xlim([0, 250])

    # plot histogram of correlations
    plot_hist_of_corr(ax[1], corr, p, bins)
    print(f"Mean correlation: {np.mean(corr):.3f}, p-value: {p:.3f}")

    # add y label
    ax[0].set_ylabel("PEARSON'S R", fontsize=16)

    # add x label
    ax[0].set_xlabel("TIME (s)", fontsize=16)


    if save_path is not None:
        plt.savefig(save_path)



def plot_corr_hist_poster(acc, save_path = None):

    # only visual
    vis_indices = [0, 1, 2, 3, 4, 5, 6]
    vis_acc = acc[vis_indices, :, :, :][:, vis_indices, :, :]

    order = np.array([0, 1, 2, 3, 8, 9, 10, 4, 5, 6, 7])
    dates = to_datetime(['08-10-2020', '09-10-2020', '15-10-2020', '16-10-2020', '02-03-2021', '16-03-2021', '18-03-2021', '22-10-2020', '29-10-2020', '12-11-2020', '13-11-2020'], format='%d-%m-%Y')

       
    # set up figure
    gs_kw = dict(width_ratios=[1, 0.4], wspace=0.01, hspace=0.3)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=300, gridspec_kw=gs_kw, sharey=True)

    bin_range = (-0.65, 0.65)

    bins = np.linspace(bin_range[0], bin_range[1], 20)

    tmp_dates = dates.copy()[vis_indices]


    dist = get_distance_matrix(tmp_dates)

    ax_hist = axes[1]
    ax_corr = axes[0]

    # get x and y
    X, y = prep_x_y(vis_acc, dist)

    # get correlation and p-values
    corr, pval = get_corr_pval(X, y)

    # test if mean of correlations is significantly different from 0
    t, p = ttest_1samp(corr, 0)


    # plot correlation
    ax_corr.plot(corr) 
            
    # seconds on x axis
    x_axis_seconds(ax_corr)
            
    # set limits
    ax_corr.set_xlim([0, 250])

    # plot histogram of correlations
    plot_hist_of_corr(ax_hist, corr, p, bins)


    axes[0].set_xlabel("Time (s)", fontsize=16)
    axes[0].set_ylabel("Pearson's R", fontsize=16)

    
    # first column y label
    #axes[0, 0].set_ylabel("Visual".upper())
    #axes[1, 0].set_ylabel("Memory".upper())
    #axes[2, 0].set_ylabel("Combined".upper())

    # share title between columns
    #axes[0, 0].set_title("Days".upper())
    #axes[0, 2].set_title("Sessions".upper())

    if save_path is not None:
        plt.savefig(save_path )


def main():
    path = Path(__file__)

    # load accuracies from cross decoding
    acc = np.load(path.parents[0] / "accuracies" / f"cross_decoding_10_LDA_sens.npy", allow_pickle=True)

    # output path
    plot_path = path.parents[0] / "plots" 

    # plot
    #plot_corr_hist(acc = acc, save_path = plot_path / "corr_acc_dist.png")

    #plot_corr_hist_cond(acc, save_path=plot_path / "corr_acc_dist_cond.png")

    # plot for poster with only visual days
    plot_corr_hist_poster(acc = acc, save_path = plot_path / "corr_acc_dist_poster.png")

        

if __name__ == "__main__":
    main()
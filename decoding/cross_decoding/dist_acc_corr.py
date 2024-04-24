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
from mne.stats import permutation_cluster_test

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

def permute_x_y(X, y, n_perm, correlation=pearsonr, stratified_permutation=False):
     # array for storing actual correlation
    corrs = np.zeros(X.shape[1])

    if stratified_permutation:
        # get unique y values
        y_unique = np.unique(y)
        
        # get n_perm permutations
        perms = [np.random.permutation(y_unique) for _ in range(n_perm)]

    # initialize numpy array to store all permutations
    permutations_corr = np.zeros((n_perm, len(X[0])))

    # loop over time points
    for t in range(X.shape[1]):
        # actual correlation
        corrs[t], _ = correlation(X[:, t], y)

    
    # loop over all permutations
    for n in range(n_perm):
        if not stratified_permutation:
        # permute y
            perm_y = np.random.permutation(y)
        else:
            perm_y = y.copy()

            # replace y values with permuted values
            for idx, y_val in enumerate(perm_y):
                # get the index of the permuted value
                perm_idx = np.where(y_unique == y_val)[0][0]
                perm_y[idx] = perms[n][perm_idx]
            
        for t in range(X.shape[1]):
            # calculate correlation
            perm_corr, _ = pearsonr(X[:, t], perm_y)

            # store permutation correlation
            permutations_corr[n, t] = perm_corr

    print(permutations_corr)
    return corrs, permutations_corr


def permutation_test(X, y, n_perm):
    """
    Permutation test to see if correlation is significant using cluster permutation test

    Parameters
    ----------
    X : numpy array
        Array of shape (n_sessionpairs, n_timepoints) containing the decoding accuracies
    """
    corrs, permutations_corr = permute_x_y(X, y, n_perm)

    pvals = np.zeros(X.shape[1])
    
    # calculate p-value
    for t in range(X.shape[1]):
        pvals[t] = np.sum(abs(permutations_corr[:, t]) > abs(corrs[t])) / n_perm
        
    return corrs, permutations_corr, pvals

def cluster_permutation_test_mne(X, y, n_perm, correlation=pearsonr, stratified_permutation=False):
    """
    Uses the MNE python function to conduct a cluster permutation test
    """
    print("Starting cluster permutation test by creating the permutations")
    corrs, permutations_corr = permute_x_y(X, y, n_perm, correlation = correlation, stratified_permutation=stratified_permutation)

    print("corrs shape", corrs.shape)
    print("permutations_corr shape", permutations_corr.shape)
    # reshape corrs
    corrs = corrs.reshape(1, -1)
    
    # calculate p-value
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [corrs, permutations_corr], 
        n_permutations=n_perm, 
        threshold=0.05)
    
    return corrs, permutations_corr, clusters, cluster_p_values



def plot_corr_hist(acc, save_path = None, corr_color="C0", perm_color="lightblue", alpha=0.05, cluster=False, n_perm=1000, correlation=pearsonr, distance = "days", stratified_permutation=False):

    # convert to datetime
    dates = to_datetime(['08-10-2020', '09-10-2020', '15-10-2020', '16-10-2020', '02-03-2021', '16-03-2021', '18-03-2021'], format='%d-%m-%Y')
       
    # set up figure
    print("Setting up figure")
    gs_kw = dict(width_ratios=[1, 0.3], height_ratios=[1], wspace=0.01, hspace=0.3)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300, gridspec_kw=gs_kw, sharey=True)

    bin_range = (-0.65, 0.65)

    bins = np.linspace(bin_range[0], bin_range[1], 20)

    if distance == "days":
        days = dates.copy()
        dist = get_distance_matrix(days)
        
    elif distance == "session":
        sessions = np.arange(7)
        dist = get_distance_matrix(sessions)


    ax_hist = axes[1]
    ax_corr = axes[0]

    # get x and y
    X, y = prep_x_y(acc, dist)
    del acc
    print(f"Done preparing x and y, with the following shapes: {X.shape}, {y.shape}")


    # permutation test to see if correlation is significant
    if not cluster:
        print("Starting permutation test")
        corr, all_perm, pvals = permutation_test(X, y, n_perm)
        print("Done with permutation test")
    else:
        print("Starting cluster permutation test")
        corr, all_perm, clusters, pvals_tmp = cluster_permutation_test_mne(X, y, n_perm, correlation=correlation, stratified_permutation=stratified_permutation)
        print("Done with cluster permutation test")
        # reshape corr
        corr = corr.reshape(-1)

        # get indices of significant clusters
        sig_clusters_idx = np.where(pvals_tmp < alpha)[0]

        # get indices of significant time points
        try: 
            sig_timepoints = np.concatenate([clusters[i][0] for i in sig_clusters_idx])
        except:
            sig_timepoints = []
            
        # get pvals
        pvals = np.ones(X.shape[1])

        # set pvals of significant time points to 0
        pvals[sig_timepoints] = 0
            
    for perm in all_perm:
        ax_corr.plot(perm, color=perm_color, linewidth=0.5, alpha=0.4)

    print("Plotting correlation and histogram")
    # plot correlation
    plot_corr(ax_corr, corr, pvals, alpha = alpha, color=corr_color, y_lim=(-1, 1))

    # get y limits
    y_lim = ax_corr.get_ylim()

    # plot histogram of correlations
    plot_hist_of_corr(ax_hist, corr, bins, color=corr_color, y_lim=y_lim)


    fig.supxlabel("TIME (s)", fontsize=16)
    fig.supylabel("Pearson's R", fontsize=16)


    if save_path is not None:
        plt.savefig(save_path)

if __name__ == "__main__":
    path = Path(__file__)

    # load accuracies from cross decoding
    acc = np.load(path.parents[0] / "accuracies" / f"cross_decoding_10_LDA_sens.npy", allow_pickle=True)

    acc = acc[:7, :7, ...]

    # output path
    plot_path = path.parents[0] / "plots" 

    alpha = 0.05

    distance = "days"
    plot_corr_hist(
        acc = acc, 
        distance=distance,
        save_path = plot_path / f"corr_acc_dist_{distance}_strat.png",
        alpha = alpha,
        cluster=True,
        stratified_permutation=True
        )

    plot_corr_hist(
        acc = acc, 
        distance=distance,
        save_path = plot_path / f"corr_acc_dist_{distance}.png",
        alpha = alpha,
        cluster=True)
    

    distance = "session"

        
    plot_corr_hist(
        acc = acc, 
        distance=distance,
        save_path = plot_path / f"corr_acc_dist_{distance}_strat.png",
        alpha = alpha,
        cluster=True,
        stratified_permutation=True
        )

    plot_corr_hist(
        acc = acc, 
        distance=distance,
        save_path = plot_path / f"corr_acc_dist_{distance}.png",
        alpha = alpha,
        cluster=True)  

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

# local imports
import sys
sys.path.append(str(Path(__file__).parents[1] / "cross_decoding"))

from dist_acc_corr import get_distance_matrix, prep_x_y, get_corr_pval, plot_corr

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
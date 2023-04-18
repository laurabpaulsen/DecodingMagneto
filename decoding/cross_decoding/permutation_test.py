"""
This script performs a permutation test on the cross decoding results.

dev notes:
- for now only testing the difference between mem_mem and vis_vis
- maybe move these functions to utils.analysis??
- add possiblility to not use multiprocessing

"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from utils.analysis.permute import tgm_permutation

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

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

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument('-p', '--parcellation', type=str, help='Parcellation to use.', default="sens")
    ap.add_argument('--train1', type=str, help='Training condition of the first group.', default="mem")
    ap.add_argument('--test1', type=str, help='Testing condition of the first group.', default="mem")
    ap.add_argument('--train2', type=str, help='Training condition of the second group.', default="vis")
    ap.add_argument('--test2', type=str, help='Testing condition of the second group.', default="vis")

    return ap.parse_args()

def load_acc(parcellation):
    """
    Loads the accuracies from a file and sets the accuracies of the training and testing on the same session to nan.

    Parameters
    ----------
    file : str
        Name of the file to load.

    Returns
    -------
    acc : numpy.ndarray
        Array containing the accuracies.
    """
    acc = np.load(os.path.join('accuracies', f"cross_decoding_10_LDA_{parcellation}.npy"), allow_pickle=True)

    # setting training and testing on the same session to nan
    acc[np.arange(acc.shape[0]), np.arange(acc.shape[1]), :, :] = np.nan
    
    return acc

def statistic(a, b, axis=0):
    """
    Calculates the statistic for the permutation test.

    Parameters
    ----------
    a : numpy.ndarray
        Array containing the accuracies of the first group.
    b : numpy.ndarray
        Array containing the accuracies of the second group.
    axis : int
        Axis along which the mean is calculated. Default is 0.

    Returns
    -------
    statistic : float
        The statistic for the permutation test.
    """
    return np.mean(a, axis=axis) - np.mean(b, axis=axis)

def plot_values(array, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(array, cmap='Reds_r', origin='lower', vmin=0, vmax=0.20)
    # contour showing the significance
    ax.contour(array, levels=[0.05], colors='k', linewidths=1, origin='lower')

    ax.set_xlabel('Testing time')
    ax.set_ylabel('Training time')
    fig.suptitle('Permutation test', fontsize=20)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('P-value', rotation=-90, va="bottom")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)


def main():
    args = parse_args()
    acc = load_acc(args.parcellation)

    vis = np.array([0, 1, 2, 3, 4, 5, 6])
    mem = np.array([7, 8, 9, 10])

    train_1 = vis if args.train1 == "vis" else mem
    test_1 = vis if args.test1 == "vis" else mem
    train_2 = vis if args.train2 == "vis" else mem
    test_2 = vis if args.test2 == "vis" else mem

    acc1 = acc[train_1, :, :, :][:, test_1, :, :]
    acc2 = acc[train_2, :, :, :][:, test_2, :, :]

    p_values = tgm_permutation(acc1, acc2, statistic, n_jobs=4)

    # save the p-values and difference in statistics
    np.save(os.path.join('permutation_results', f"{args.parcellation}_p_values_{args.train1}{args.test1}_{args.train2}{args.test2}.npy"), p_values)
    
    # plot the p-values
    plot_values(p_values, save_path = os.path.join("permutation_results", f"{args.parcellation}_p_values_{args.train1}{args.test1}_{args.train2}{args.test2}.png"))


if __name__ == '__main__':
    main()
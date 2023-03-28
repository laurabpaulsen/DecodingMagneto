"""
This script performs a permutation test on the cross decoding results.

dev notes:
- for now only testing the difference between mem_mem and vis_vis
- maybe move these functions to utils.analysis??

"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from utils.analysis import plot
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
from tqdm import tqdm
import argparse
import multiprocessing as mp


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

    ap.add_argument('-p', '--parcellation', type=str, help='Parcellation to use.', default="HCPMMP1")

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

def prep_X_y_permute(array, label):
    X = array.flatten()
    X = X[~np.isnan(X)]

    y = np.array([label]*len(X))

    return X, y

def statistic(a, b):
    """
    Calculates the statistic for the permutation test.

    Parameters
    ----------
    a : numpy.ndarray
        Array containing the accuracies of the first group.
    b : numpy.ndarray
        Array containing the accuracies of the second group.

    Returns
    -------
    statistic : float
        The statistic for the permutation test.
    """
    return np.mean(a) - np.mean(b)

def tgm_permutation(acc1, acc2, statistic, n = 10):
    """
    Performs a permutation test on the temporal generalisation matrices with the cross decoding results.

    Returns
    -------
    p_values : numpy.ndarray
        Array containing the p-values for each region.
    
    diff_statistic : numpy.ndarray
        Array containing the difference between the true statistic and the permuted statistic.
    """

    # check that the temporal generalisation matrices have the same shape
    assert acc1[0, 0].shape == acc2[0, 0].shape, "The temporal generalisation matrices have different shapes."

    n_time_points = 250//n

    # empty array to store the p-values
    p_values = np.zeros((n_time_points, n_time_points))

    # empty array to store the difference between the statistic and the permuted statistic
    diff_stats = p_values.copy()

    pool = mp.Pool(mp.cpu_count()-4)

    for i in tqdm(range(n_time_points)): # loop over training time points
        i_ind = get_indices(i, n)

        # loop over testing time points in parallel
        results = pool.starmap(permutation, [(acc1[:, :, i_ind, get_indices(j, n)], acc2[:, :, i_ind, get_indices(j, n)], statistic) for j in range(n_time_points)])

        for j, result in enumerate(results):
            p_values[i, j] = result[0]
            diff_stats[i, j] = result[1]

    return p_values, diff_stats


def get_indices(i, n):
    i_ind = [n*i+add for add in range(n)]


    return i_ind


def permutation(acc1, acc2, statistic):
    """
    Performs a permutation test on two arrays.

    Parameters
    ----------
    acc1 : numpy.ndarray
        Array containing the accuracies of the first group.
    acc2 : numpy.ndarray
        Array containing the accuracies of the second group.
    statistic : function
        Function that calculates the statistic for the permutation test.
    
    Returns
    -------
    p_value : float
        The p-value for the permutation test.
    
    diff_statistic : float
        The difference between the true statistic and the permuted statistic.
    """   
    acc1_tmp, acc1_y = prep_X_y_permute(acc1, 0)
    acc2_tmp, acc2_y = prep_X_y_permute(acc2, 1)

    unpermuted = statistic(acc1_tmp, acc2_tmp)

    # concatenate the accuracies and labels
    X = np.concatenate((acc1_tmp, acc2_tmp))
    y = np.concatenate((acc1_y, acc2_y))

    # permutation test
    result = sp.stats.permutation_test((X, y), statistic=statistic)

    diff_statistic = unpermuted - result.statistic
    p_value = result.pvalue

    return p_value, diff_statistic

def plot_values(array, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(array, cmap='Reds', origin='lower', interpolation = None)

    ax.set_xlabel('Testing time')
    ax.set_ylabel('Training time')
    fig.suptitle('Permutation test')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Difference in true mean accuracy compared to difference in permuted accuracy', rotation=-90, va="bottom")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)


def main():
    args = parse_args()
    acc = load_acc(args.parcellation)

    # all trained on memory, tested on memory
    mem_mem = acc[7:, 7:]

    # all trained on vis and tested on vis
    vis_vis = acc[:7, :7]

    # how many timepoints to combine during the permutation test (e.g. n = 5 means that 5 timepoints are combined into one)
    n_time = 5

    p_values, diff_stats = tgm_permutation(mem_mem, vis_vis, statistic, n=n_time)

    # save the p-values and difference in statistics
    np.save(os.path.join('permutation_results', f"p_values_{args.parcellation}.npy"), p_values)
    np.save(os.path.join('permutation_results', f"diff_stats_{args.parcellation}.npy"), diff_stats)

    # plot the difference in statistics
    plot_values(diff_stats, save_path = os.path.join('permutation_results', f"diff_stats_{args.parcellation}.png"))

    # plot the p-values
    plot_values(p_values, save_path = os.path.join('permutation_results', f"p_values_{args.parcellation}.png"))


if __name__ == '__main__':
    main()



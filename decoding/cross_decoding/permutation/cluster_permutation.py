"""
Script for conducting cluster-based permutation tests on cross-decoding results.
"""

import argparse
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from mne.stats import permutation_cluster_test

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument('-p', '--parcellation', type=str, help='Parcellation to use.', default="sens")
    ap.add_argument('--train1', type=str, help='Training condition of the first group.', default="mem")
    ap.add_argument('--test1', type=str, help='Testing condition of the first group.', default="mem")
    ap.add_argument('--train2', type=str, help='Training condition of the second group.', default="vis")
    ap.add_argument('--test2', type=str, help='Testing condition of the second group.', default="vis")

    return ap.parse_args()

def load_acc(path):
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
    acc = np.load(path, allow_pickle=True)

    # setting training and testing on the same session to nan
    acc[np.arange(acc.shape[0]), np.arange(acc.shape[1]), :, :] = np.nan
    
    return acc

def divide_acc(acc, args, vis_idx: np.array = np.array([0, 1, 2, 3, 4, 5, 6]), mem_idx: np.array = np.array([7, 8, 9, 10])):

    train_1 = vis_idx if args.train1 == "vis" else mem_idx
    test_1 = vis_idx if args.test1 == "vis" else mem_idx
    train_2 = vis_idx if args.train2 == "vis" else mem_idx
    test_2 = vis_idx if args.test2 == "vis" else mem_idx


    acc1 = acc[train_1, :, :, :][:, test_1, :, :]
    acc2 = acc[train_2, :, :, :][:, test_2, :, :]

    return acc1, acc2

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

def main():
    args = parse_args()

    path = Path(__file__).parents[1]

    # loading accuracies
    acc = load_acc(path /'accuracies' / f"cross_decoding_10_LDA_{args.parcellation}.npy")

    # dividing accuracies into two groups
    acc1, acc2 = divide_acc(acc, args)

    print(acc1.shape)
    print(acc2.shape)

    # reshaping accuracies
    # merge two first dimensions ignoring nan values
    acc1 = acc1.reshape(acc1.shape[0] * acc1.shape[1], acc1.shape[2], acc1.shape[3])
    acc2 = acc2.reshape(acc2.shape[0] * acc2.shape[1], acc2.shape[2], acc2.shape[3])

    # remove nan values
    acc1 = acc1[~np.isnan(acc1).any(axis=(1, 2))]
    acc2 = acc2[~np.isnan(acc2).any(axis=(1, 2))]

    print(acc1.shape)
    print(acc2.shape)



    # cluster-based permutation test
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test([acc1, acc2], n_permutations=1000, threshold=0.01, tail=0, stat_fun=statistic, n_jobs=1, verbose=True)

    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    print(f"Number of significant clusters: {len(good_cluster_inds)}")
    
    if len(good_cluster_inds) > 0:
        # empty array for storing significant clusters
        significant_clusters = np.zeros((acc1.shape[1], acc1.shape[2]))
            # filling significant clusters
        for i in good_cluster_inds:
            significant_clusters[clusters[i]] = 1

        # save array
        np.save(path / "permutation" / f"{args.parcellation}_sig_clusters_{args.train1}{args.test1}_{args.train2}{args.test2}.npy", significant_clusters)
        
        # plot image
        p = plt.imshow(significant_clusters, cmap="gray_r", origin="lower", interpolation="none")
        plt.colorbar(p)
        plt.savefig(path / "permutation" / f"{args.parcellation}_sig_clusters_{args.train1}{args.test1}_{args.train2}{args.test2}.png")

if __name__ == "__main__":
    main()

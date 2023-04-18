import multiprocessing as mp
import numpy as np
import scipy as sp
from tqdm import tqdm


def prep_X_permute(array):
    X = array.flatten()
    X = X[~np.isnan(X)]

    return X

def tgm_permutation(acc1, acc2, statistic, n_jobs=1):
    """
    Performs a permutation test on the temporal generalisation matrices with the cross decoding results.

    Parameters
    ----------
    acc1 : numpy.ndarray
        Array containing the accuracies of the first group.
    acc2 : numpy.ndarray
        Array containing the accuracies of the second group.
    statistic : function
        Function that calculates the statistic for the permutation test.
    n_jobs : int
        Number of jobs to run in parallel. Default is 1.

    Returns
    -------
    p_values : numpy.ndarray
        Array containing the p-values for each region.
    
    diff_statistic : numpy.ndarray
        Array containing the difference between the true statistic and the permuted statistic.
    """

    # check that the temporal generalisation matrices have the same shape
    assert acc1[0, 0].shape == acc2[0, 0].shape, "The temporal generalisation matrices have different shapes."

    n_time_points = acc1[0, 0].shape[0]

    # empty array to store the p-values
    p_values = np.zeros((n_time_points, n_time_points))

    pool = mp.Pool(n_jobs)

    for i in tqdm(range(n_time_points)): # loop over training time points
        # loop over testing time points in parallel
        results = pool.starmap(permutation, [(acc1[:, :, i, j], acc2[:, :, i, j], statistic) for j in range(n_time_points)])

        for j, result in enumerate(results):
            p_values[i, j] = result

    return p_values

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
    """   
    acc1_tmp = prep_X_permute(acc1)
    acc2_tmp = prep_X_permute(acc2)

    # permutation test
    result = sp.stats.permutation_test((acc1_tmp, acc2_tmp), statistic=statistic, n_resamples=1000)

    return result.pvalue

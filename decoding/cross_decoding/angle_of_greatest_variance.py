from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import ttest_1samp


def get_angle_of_greatest_variance(X):
    """
    Get the angle of greatest variance matrix X
    """

    # get PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    
    # get components
    components = pca.components_
    
    # get angle of greatest variance
    angle = np.arctan2(components[0, 1], components[0, 0])
    
    return angle

def angle_of_greatest_variance_matrix2D(X):
    """
    Get the angle of greatest variance matrix X
    """

    # get PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    
    # get components
    components = pca.components_
    # get angle of greatest variance
    angle = np.arctan2(components[0, 1], components[0, 0])
    
    return angle

def angles_from_crossdecoding(X):
    """
    Get angles from cross decoding matrix X

    Parameters
    ----------
    X : array
        Cross decoding matrix of shape (n_sessions, n_sessions, n_bins, n_bins)
    
    Returns
    -------
    angles : list
        List of angles of greatest variance
    """

    # initialize list
    angles = []

    # loop over pairs of sessions
    for i in range(X.shape[0]):
        tmp = X[i, :, :]

        for j in range(X.shape[1]):
            tmp_acc = tmp[j, :, :]

            # get angle of greatest variance
            angle = get_angle_of_greatest_variance(tmp_acc)
            angles.append(angle)

    return angles

if __name__ in "__main__":
    path = Path(__file__)

    # load accuracies
    acc = np.load(path.parents[0] / "accuracies" / f"cross_decoding_10_LDA_sens.npy", allow_pickle=True)
    
    vis_index = [0, 1, 2, 3, 4, 5, 6]
    mem_index = [7, 8, 9, 10]

    # get accuracies for testing on visual and training on memory
    acc_subset_vis_mem = acc[vis_index, :, :, :][:, mem_index, :, :]
    acc_subset_mem_vis = acc[mem_index, :, :, :][:, vis_index, :, :]

    for acc_subset, cond in zip([acc_subset_vis_mem, acc_subset_mem_vis], ["vis_mem", "mem_vis"]):
        # get angles
        angles = angles_from_crossdecoding(acc_subset)

        t, p = ttest_1samp(angles, 0)
        print(cond)
        print(f"Mean angle: {np.mean(angles):.3f}, p-value: {p:.3f}")


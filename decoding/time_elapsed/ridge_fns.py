"""
Functions for ridge regression
"""
import argparse as ap
import numpy as np
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

def fit_ridge_clf(X, y, alphas = [1e-3, 1e-2, 1e-1, 1], ncv = 10):
    """
    Fits the ridge classifier to the data.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    y : np.ndarray
        Target vector.
    alphas : list
        List of alpha values to try.
    ncv : int
        Number of cross validation folds to use when fitting the classifier.
    
    Returns
    -------
    clf : sklearn.linear_model.Ridge
        Fitted ridge classifier.
    """

    # create the classifier
    clf = RidgeCV(alphas = alphas, cv = ncv)

    # fit the classifier
    clf.fit(X, y)

    return clf

def test_ridge_clf(clf, X, y):
    """
    Tests the ridge classifier on the data.
    
    Parameters
    ----------
    clf : sklearn.linear_model.Ridge
        Fitted ridge classifier.
    X : np.ndarray
        Data matrix.
    y : np.ndarray
        Target vector.
    
    Returns
    -------
    score : float
        Mean squared error.
    """

    # get the score
    score = clf.score(X, y)

    return score

def tgm_ridge_scores(X, y, stratify, alphas = [1e-17, 1e-20, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1], ncv = 10):
    """
    Uses ridge regression to create a temporal generalisation matrix of scores. 

    Parameters
    ----------
    X : np.ndarray 
        Data matrix, shape (n_timepoints, n_trials, n_features).
    y : np.ndarray
        Target vector.
    stratify : np.ndarray
        Stratification vector. Default is None. This vector is used to ensure no leakage between training and testing sets by leaving out one stratification group at a time.
    alphas : list
        List of alpha values to try.
    ncv : int
        Number of cross validation folds to use when fitting the classifier.

    Returns
    -------
    tgm : np.ndarray
        Temporal generalisation matrix of scores, shape (n_timepoints, n_timepoints).
    """

    # get the number of timepoints
    n_timepoints = X.shape[0]

    # create the temporal generalisation matrix
    tgm = np.zeros((n_timepoints, n_timepoints, len(np.unique(stratify))))

    # loop over the timepoints
    for i in tqdm(range(n_timepoints)):
        # loop over stratification groups
        for s, strat in enumerate(np.unique(stratify)):
            # get the training and testing data
            X_train = X[i, stratify != strat, :]
            y_train = y[stratify != strat]
            X_test = X[:, stratify == strat, :]
            y_test = y[stratify == strat]

            # shuffle the training data
            idx = np.random.permutation(X_train.shape[0])
            X_train = X_train[idx, :]
            y_train = y_train[idx]

            # fit the classifier
            clf = fit_ridge_clf(X_train, y_train, alphas = alphas, ncv = ncv)

            # print the best alpha
            print("best alpha", clf.alpha_)

            # test the classifier on all timepoint
            for j in range(n_timepoints):
                # get the testing data
                X_test_j = X_test[j, :, :]

                # predict the session number
                pred = clf.predict(X_test_j)

                mean_error = np.mean(np.abs(pred - y_test))

                # store the score
                tgm[i, j, s] = mean_error
    
    return tgm

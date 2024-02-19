"""
Functions for ridge regression used in predict_session_day.py, predict_trial_number.py, and predict_session_number.py.
"""
import numpy as np
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
 from sklearn.model_selection import KFold

import logging

def get_logger(filename):
    logger = logging.getLogger()  # Get a logger instance
    logger.setLevel(logging.INFO)  # Set logging level

    # Create a file handler
    file_handler = logging.FileHandler(filename, mode='w')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger
 

def fit_ridge_clf(X, y, alphas:list, ncv = 10):
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

def fit_elasticnet_clf(X, y, alphas:list, ncv = 10):
    """
    Fits the elastic net classifier to the data.
    
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
    clf : sklearn.linear_model.ElasticNet
        Fitted ridge classifier.
    """

    # create the classifier
    clf = ElasticNetCV(alphas = alphas, cv = ncv)

    # fit the classifier
    clf.fit(X, y)

    return clf


def tgm_elasticnet_scores(X, y, stratify, alphas = np.logspace(0, 2, 10), ncv = 10, logger = None):
    """
    Fits the ridge classifier to the data and applies it to all timepoints.

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
    predictions : np.ndarray
        Matrix of predictions, shape (n_timepoints, n_timepoints, n_stratification_groups, n_trials).

    true_values : np.ndarray
        Matrix of true values, shape (n_timepoints, n_timepoints, n_stratification_groups, n_trials).
    """
    # get the number of timepoints
    n_timepoints = X.shape[0]

    # get the number of stratification groups
    n_stratification_groups = len(np.unique(stratify))

    # get the largest number of trials in a stratification group
    n_trials = np.max(np.unique(stratify, return_counts=True)[1])

    # create the predictions and true value matrices
    predictions = np.empty((n_timepoints, n_timepoints, n_stratification_groups, n_trials))
    true_values = np.empty((n_timepoints, n_timepoints, n_stratification_groups, n_trials))

    # fill with nans
    predictions[:] = np.nan
    true_values[:] = np.nan

    # shuffle the data
    idx = np.random.permutation(X.shape[1])
    X = X[:, idx, :]
    y = y[idx]
    stratify = stratify[idx]


    # loop over the timepoints
    for i in tqdm(range(n_timepoints)):
        # loop over stratification groups
        for s, strat in enumerate(np.unique(stratify)):
            # get the training and testing data
            X_train = X[i, stratify != strat, :]
            y_train = y[stratify != strat]
            X_test = X[:, stratify == strat, :]
            y_test = y[stratify == strat]

            # standardize the data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

            # fit the classifier
            clf = fit_elasticnet_clf(X_train, y_train, alphas = alphas, ncv = ncv)

            # test the classifier on all timepoint
            for j in range(n_timepoints):
                # get the testing data
                X_test_j = X_test[j, :, :]
                # standardize the data
                X_test_j = scaler.transform(X_test_j)

                # predict the labels
                pred = clf.predict(X_test_j)

                # store the predictions and true values
                predictions[i, j, s, :pred.shape[0]] = pred
                true_values[i, j, s, :pred.shape[0]] = y_test

                # log if all predictions made using model trained on time point i are the same for stratification group s (i.e. no variance in predictions)
                if logger is not None:
                    if np.all(pred == pred[0]):
                        logger.info(f'All predictions for train timepoint {i} and test timepoint {j} (stratgroup {s}) are the same.')

                
    return predictions, true_values


def tgm_ridge_scores(X, y, stratify, alphas = np.logspace(0, 2, 10), ncv = 10, logger = None):
    """
    Fits the ridge classifier to the data and applies it to all timepoints.

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
    predictions : np.ndarray
        Matrix of predictions, shape (n_timepoints, n_timepoints, n_stratification_groups, n_trials).

    true_values : np.ndarray
        Matrix of true values, shape (n_timepoints, n_timepoints, n_stratification_groups, n_trials).
    """
    # get the number of timepoints
    n_timepoints = X.shape[0]

    # get the number of stratification groups
    n_stratification_groups = len(np.unique(stratify))

    # get the largest number of trials in a stratification group
    n_trials = np.max(np.unique(stratify, return_counts=True)[1])

    # create the predictions and true value matrices
    predictions = np.empty((n_timepoints, n_timepoints, n_stratification_groups, n_trials))
    true_values = np.empty((n_timepoints, n_timepoints, n_stratification_groups, n_trials))

    # fill with nans
    predictions[:] = np.nan
    true_values[:] = np.nan

    # shuffle the data
    idx = np.random.permutation(X.shape[1])
    X = X[:, idx, :]
    y = y[idx]
    stratify = stratify[idx]


    # loop over the timepoints
    for i in tqdm(range(n_timepoints)):
        # loop over stratification groups
        for s, strat in enumerate(np.unique(stratify)):
            # get the training and testing data
            X_train = X[i, stratify != strat, :]
            y_train = y[stratify != strat]
            X_test = X[:, stratify == strat, :]
            y_test = y[stratify == strat]

            # standardize the data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

            # fit the classifier
            clf = fit_ridge_clf(X_train, y_train, alphas = alphas, ncv = ncv)

            # test the classifier on all timepoint
            for j in range(n_timepoints):
                # get the testing data
                X_test_j = X_test[j, :, :]
                # standardize the data
                X_test_j = scaler.transform(X_test_j)

                # predict the labels
                pred = clf.predict(X_test_j)

                # store the predictions and true values
                predictions[i, j, s, :pred.shape[0]] = pred
                true_values[i, j, s, :pred.shape[0]] = y_test

                # log if all predictions made using model trained on time point i are the same for stratification group s (i.e. no variance in predictions)
                if logger is not None:
                    if np.all(pred == pred[0]):
                        logger.info(f'All predictions for train timepoint {i} and test timepoint {j} (stratgroup {s}) are the same.')

                
    return predictions, true_values

def tgm_ridge_scores_unstratified(X, y, cv = 10, alphas = np.logspace(0, 2, 10), ncv = 10, logger = None):
    """
    Fits the ridge classifier to the data and applies it to all timepoints.

    Parameters
    ----------
    X : np.ndarray 
        Data matrix, shape (n_timepoints, n_trials, n_features).
    y : np.ndarray
        Target vector.
    cv : int
        Number of cross validation folds.
    stratify : np.ndarray
        Stratification vector. Default is None. This vector is used to ensure no leakage between training and testing sets by leaving out one stratification group at a time.
    alphas : list
        List of alpha values to try.
    ncv : int
        Number of cross validation folds determine the alpha value to use.

    Returns
    -------
    predictions : np.ndarray
        Matrix of predictions, shape (n_timepoints, n_timepoints, n_stratification_groups, n_trials).

    true_values : np.ndarray
        Matrix of true values, shape (n_timepoints, n_timepoints, n_stratification_groups, n_trials).
    """
    # get the number of timepoints
    n_timepoints = X.shape[0]

    folds = KFold(n_splits=cv, shuffle=True)

    # get the largest number of trials in a stratification group
    n_trials = np.max(np.unique(stratify, return_counts=True)[1])

    # create the predictions and true value matrices
    predictions = np.empty((n_timepoints, n_timepoints, cv, n_trials))
    true_values = np.empty((n_timepoints, n_timepoints, cv, n_trials))

    # fill with nans
    predictions[:] = np.nan
    true_values[:] = np.nan

    # shuffle the data
    idx = np.random.permutation(X.shape[1])
    X = X[:, idx, :]
    y = y[idx]
    stratify = stratify[idx]


    # loop over the timepoints
    for i in tqdm(range(n_timepoints)):
        # loop over k-folds
        for j, (train_idx, test_idx) in enumerate(folds.split(X[i, :, :])):
            # get the training and testing data
            X_train = X[i, train_idx, :]
            y_train = y[train_idx]
            X_test = X[:, test_idx, :]
            y_test = y[test_idx]

            # standardize the data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

            # fit the classifier
            clf = fit_ridge_clf(X_train, y_train, alphas = alphas, ncv = ncv)

            # test the classifier on all timepoint
            for j in range(n_timepoints):
                # get the testing data
                X_test_j = X_test[j, :, :]
                # standardize the data
                X_test_j = scaler.transform(X_test_j)

                # predict the labels
                pred = clf.predict(X_test_j)

                # store the predictions and true values
                predictions[i, j, j, :pred.shape[0]] = pred
                true_values[i, j, j, :pred.shape[0]] = y_test

                # log if all predictions made using model trained on time point i are the same for stratification group s (i.e. no variance in predictions)
                if logger is not None:
                    if np.all(pred == pred[0]):
                        logger.info(f'All predictions for train timepoint {i} and test timepoint {j} (stratgroup {s}) are the same.')

                
    return predictions, true_values
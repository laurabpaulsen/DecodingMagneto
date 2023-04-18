import numpy as np

def n_trials(X, y, n: int, axis: int=1):
    """
    Removes trials from X and y, such that the number of trials is equal to n. It is assumed that classes are already balanced. Therefore an equal number of trials is removed from each class.

    Parameters:
    ----------
    X : np.array
        data
    y : np.array
        labels (0 or 1)
    n : int
        number of trials to keep
    axis : int, optional
        axis along which to remove trials from X array, by default 1
    
    Returns:
    -------
    X : np.array
        data with n trials
    y : np.array
        labels with n trials
    """

    # total number of trials
    n_trials = len(y)

    # number of trials to remove per condition
    n_remove = (n_trials - n)//2

    # getting indices of trials to remove
    idx_0 = np.random.choice(np.where(y==0)[0], n_remove, replace=False)
    idx_1 = np.random.choice(np.where(y==1)[0], n_remove, replace=False)

    # combining indices
    idx = np.concatenate((idx_0, idx_1))

    # removing trials
    X = np.delete(X, idx, axis=axis)
    y = np.delete(y, idx)

    return X, y

def equalise_trials(Xs:list, ys:list):
    """
    This function is used to equalise the number of trials across a list of X and y arrays.

    Parameters
    ----------
    Xs : list
        list of X arrays
    ys : list
        list of y arrays
    
    Returns
    -------
    Xs : list
        list of X arrays with equal number of trials
    ys : list
        list of y arrays with equal number of trials
    """
    # min number of trials
    min_trials = min([len(y) for y in ys])

    # make sure all sessions have the same number of trials
    for i,(X, y) in enumerate(zip(Xs, ys)):
        Xs[i], ys[i] = n_trials(X, y, min_trials)

    return Xs, ys
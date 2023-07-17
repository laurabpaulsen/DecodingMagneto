import numpy as np
import os
import json
from pathlib import Path

def get_triggers_equal():
    """
    Returns:
    -------
    triggers : list
        List of triggers for with an equal amout for animate and inanimate objects.
    
    """
    with open(os.path.join('..', '..', 'info_files', 'event_ids.txt'), 'r') as f:
        file = f.read()
        event_ids = json.loads(file)

    animate_triggers = [value for key, value in event_ids.items() if 'Animate' in key]
    inanimate_triggers = [value for key, value in event_ids.items() if 'Inanimate' in key][:len(animate_triggers)]
    
    triggers = animate_triggers.copy()
    triggers.extend(inanimate_triggers)

    return triggers

def convert_triggers_animate_inanimate(y):
    """
    Converts from image triggers to animate/inanimate triggers. If animate, y = 1, else y = 0.

    Parameters
    ----------
    y : list or array of image triggers

    Returns
    -------
    y : list or array of animate/inanimate triggers (1 or 0)
    """
    path = Path(__file__)

    with open(path.parents[2] / 'info_files' / 'event_ids.txt', 'r') as f:
        file = f.read()
        event_ids = json.loads(file)

    animate_triggers = [value for key, value in event_ids.items() if 'Animate' in key]

    y = np.array([1 if i in animate_triggers else 0 for i in y])

    return y


def balance_class_weights(X, y, verbose = True):
    """
    Balances the class weight by removing trials from the class with the most trials.

    Parameters
    ----------
    X : array
        Data array with shape (n_channels, n_trials, n_times)
    y : array
        Array with shape (n_trials, ) containing two classes (0 or 1)
    verbose : bool, optional
        Print statements. The default is True.
    
    Returns
    -------
    X_equal : array
        Data array with shape (n_channels, n_trials, n_times) with equal number of trials for each class
    y_equal : array
        Array with shape (n_trials, ) containing classes with equal number of trials for each class
    remove_ind : list
        List of indices that were removed from the data
    """
    keys, counts = np.unique(y, return_counts = True)
    if counts[0]-counts[1] > 0:
        index_inanimate = np.where(np.array(y) == 0)
        random_choices = np.random.choice(len(index_inanimate[0]), size = counts[0]-counts[1], replace=False)
        remove_ind = [index_inanimate[0][i] for i in random_choices]
    else:
        index_animate = np.where(np.array(y) == 1)
        random_choices = np.random.choice(len(index_animate[0]), size = counts[1]-counts[0], replace=False)
        remove_ind = [index_animate[0][i] for i in random_choices]

    X_equal = np.delete(X, remove_ind, axis = 1)
    y_equal = np.delete(y, remove_ind, axis = 0)

    if verbose:
        print(f'Removed a total of {len(remove_ind)} trials')
        print(f'{len(y_equal)} remains')
    
    return X_equal, y_equal, remove_ind

def balance_class_weights_multiple(X, y):
    """
    Balances the class weight by removing trials so each class has the same number of trials as the class with the least trials.

    Parameters
    ----------
    X : array
        Data array with shape (n_channels, n_trials, n_times)
    y : array
        Array with shape (n_trials, ) containing several classes

    Returns
    -------
    X_equal : array
        Data array with shape (n_channels, n_trials, n_times) with equal number of trials for each class
    y_equal : array
        Array with shape (n_trials, ) containing classes with equal number of trials for each class

    """
    keys, counts = np.unique(y, return_counts = True)

    keep_inds = []

    for key in keys:
        index = np.where(np.array(y) == key)
        random_choices = np.random.choice(index[0], size = counts.min(), replace=False)
        keep_inds.extend(random_choices)
    
    X_equal = X[:, keep_inds, :]
    y_equal = y[keep_inds]

    return X_equal, y_equal

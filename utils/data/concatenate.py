import mne
import numpy as np
import os
import pickle as pkl

def flip_sign(X1, X2):
    """
    This function is used to flip the sign of the data in X2 if the correlation between the data in X1 and X2 is negative.

    Parameters
    ----------
    X1 (array): 
        Data from the session to compare to with shape (n_channels, n_trials, n_times)
        
    X2 (array): 
        Data from the session to flip the sign of with shape (n_channels, n_trials, n_times)

    Returns
    -------
    X2 (ndarray): 
        X2 with the sign flipped if the correlation between X1 and X2 is negative in the given parcel with shape (n_channels, n_trials, n_times)
    """

    # checking that the T and P dimensions are the same
    if X1.shape[0] != X2.shape[0]:
        raise ValueError('The number of time points in the two sessions are not the same')
    
    if X1.shape[2] != X2.shape[2]:
        raise ValueError('The number of parcels in the two sessions are not the same')

    # loop over parcels
    for i in range(X1.shape[2]):
        # take means over trials
        mean1 = np.mean(X1[:, :, i], axis = 1)
        mean2 = np.mean(X2[:, :, i], axis = 1)

        # calculate correlation
        corr = np.corrcoef(mean1, mean2)[0, 1]

        if corr < 0: # if correlation is negative, flip sign
            X2[:, :, i] = X2[:, :, i] * -1

    return X2


def read_and_concate_sessions(session_files, trigger_list):
    """
    Reads and concatenates epochs from different sessions into a single epochs object.

    Parameters
    ----------
    session_files : list
        List of files to be concatenated.

    trigger_list : list
        List of triggers to be included in the concatenated epochs object.

    Returns
    -------
    X : concatenated trials from sessions
    y : concatenated labels from sessions
    """
    for i in session_files:
        epochs = mne.read_epochs(f'/media/8.1/laurap/final_data/epochs/{i}')
        if i == session_files[0]:
            y = epochs.events[:, 2]
            idx = [i for i, x in enumerate(y) if x in trigger_list]

            X = epochs.get_data(picks = 'meg')
            X = X.transpose(2, 0, 1)
            X = X[:, idx, :]

            # y
            y = np.array(y)[idx]

        else:
            y_tmp = epochs.events[:, 2]
            idx = [i for i, x in enumerate(y_tmp) if x in trigger_list]
            y_tmp = np.array(y_tmp)[idx]
            
            X_tmp = epochs.get_data(picks = 'meg')
            X_tmp = X_tmp.transpose(2, 0, 1)
            X_tmp = X_tmp[:, idx, :]

            X = np.concatenate((X, X_tmp), axis = 1)

            y = np.concatenate((y, y_tmp))
    
    return X, y

def read_and_concate_sessions_source(path, event_path, session_files, trigger_list, sign_flip = True):
    """
    Reads and concatenates epochs from different sessions into a single epochs object.
    
    Parameters
    ----------
    session_files : list
        List of session files to be concatenated.
    trigger_list : list
        List of triggers to be included in the concatenated epochs object.
    sign_flip : bool (default: True)
        Whether to flip the sign of the data in flip_session if the correlation between the data in compare_session and flip_session is negative.
    
    Returns
    -------
    X : concatenated trials from sessions
    y : concatenated labels from sessions
    """

    for sesh in session_files:

        data = np.load(os.path.join(path, f'{sesh}_parcelled.npy'))

        with open(os.path.join(event_path, f'{sesh}_events.pkl'), 'rb') as f:
            events = pkl.load(f)

        if sesh == session_files[0]:
            X = data
            y = events
            
            idx = [i for i, x in enumerate(y) if x in trigger_list]
            X = X[idx, :, :]
            X = X.transpose(2, 0, 1)

            # y
            y = np.array(y)[idx]

        else:
            y_tmp = events
            idx = [i for i, x in enumerate(y_tmp) if x in trigger_list]

            y_tmp = np.array(y_tmp)[idx]
            
            X_tmp = data
            X_tmp = X_tmp[idx, :, :]
            X_tmp = X_tmp.transpose(2, 0, 1)

            if sign_flip:
                X_tmp = flip_sign(X, X_tmp)

            X = np.concatenate((X, X_tmp), axis = 1)

            y = np.concatenate((y, y_tmp))
    
    return X, y

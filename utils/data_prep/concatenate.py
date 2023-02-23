import mne
import numpy as np
import os
import pickle as pkl

def flip_sign(X_compare_session, X_flip_session):
    """
    This function is used to flip the sign of the data in flip_session if the correlation between the data in compare_session and flip_session is negative.

    Args:
        compare_session (int): session to compare with
        flip_session (int): session to flip sign of
    """
    # loop over parcels
    for i in range(X_compare_session.shape[2]):
        # take means over trials
        mean1 = np.mean(X_compare_session[:, :, i], axis = 1)
        mean2 = np.mean(X_flip_session[:, :, i], axis = 1)

        # calculate correlation
        corr = np.corrcoef(mean1, mean2)[0, 1]
        if corr < 0:
            X_flip_session[:, :, i] = X_flip_session[:, :, i] * -1

    return X_flip_session


def read_and_concate_sessions(session_files, trigger_list):
    """Reads and concatenates epochs from different sessions into a single epochs object.
    Parameters
    ----------
    session_files : list
        List of session files to be concatenated.
    trigger_list : list
        List of triggers to be included in the concatenated epochs object.

    Returns
    -------
    X : concatenated trials from sessions
    y : concatenated labels from sessions
    """
    for i in session_files:
        epochs = mne.read_epochs(f'/media/8.1/final_data/laurap/epochs/{i}')
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

def read_and_concate_sessions_source(path, event_path, session_files, trigger_list, flip_sign = True):
    """Reads and concatenates epochs from different sessions into a single epochs object.
    Parameters
    ----------
    session_files : list
        List of session files to be concatenated.
    trigger_list : list
        List of triggers to be included in the concatenated epochs object.
    flip_sign : bool (default: True)
        Whether to flip the sign of the data in flip_session if the correlation between the data in compare_session and flip_session is negative.
    
    Returns
    -------
    X : concatenated trials from sessions
    y : concatenated labels from sessions
    """
    for file in session_files:
        data = np.load(os.path.join(path, f'{file}_parcelled.npy'))

        with open(os.path.join(event_path, f'{file}_events.pkl'), 'rb') as f:
            events = pkl.load(f)


        if file == session_files[0]:
            X = data
            y = events
            
            idx = [i for i, x in enumerate(y) if x in trigger_list]
            X = X[idx, :, :]

            # y
            y = np.array(y)[idx]

        else:
            y_tmp = events
            idx = [i for i, x in enumerate(y_tmp) if x in trigger_list]

            y_tmp = np.array(y_tmp)[idx]
            
            X_tmp = data
            X_tmp = X_tmp[idx, :, :]

            if flip_sign:
                X_tmp = flip_sign(X, X_tmp)
                print("sign flipped")

            X = np.concatenate((X, X_tmp), axis = 0)
            print(X.shape)

            y = np.concatenate((y, y_tmp))
    
        return X.transpose(2, 0, 1), y

import numpy as np
import mne
import os
import pickle as pkl
import json

def balance_class_weights(X, y):
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

    print(f'Removed a total of {len(remove_ind)} trials')
    print(f'{len(y_equal)} remains')
    
    return X_equal, y_equal, remove_ind

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

def read_and_concate_sessions_source(path, event_path, session_files, trigger_list):
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

            X = np.concatenate((X, X_tmp), axis = 0)
            print(X.shape)

            y = np.concatenate((y, y_tmp))
    
        return X.transpose(2, 0, 1), y


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
    with open(os.path.join('..', '..', 'info_files', 'event_ids.txt'), 'r') as f:
        file = f.read()
        event_ids = json.loads(file)

    animate_triggers = [value for key, value in event_ids.items() if 'Animate' in key]

    y = np.array([1 if i in animate_triggers else 0 for i in y])

    return y
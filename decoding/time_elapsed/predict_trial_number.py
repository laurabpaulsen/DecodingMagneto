
from pathlib import Path
import argparse as ap
import numpy as np
import json

# local imports
import sys
sys.path.append(str(Path(__file__).parents[2])) # adds the parent directory to the path so that the utils module can be imported
from utils.data.concatenate import read_and_concate_sessions
from ridge_fns import tgm_ridge_scores


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--trial_type', type=str, default='animate', help='Trial type. Can be either animate or inanimate.')
    parser.add_argument('--task', type=str, default='combined', help='Task. Can be either combined, visual or memory.')
    parser.add_argument('--ncv', type=int, default=10, help='Number of cross validation folds.')

    args = parser.parse_args()
    
    return args

def prepare_data(session_list:list, triggers:list):
    """
    Prepares data for decoding by concatenating sessions from different days. The y array is the session number.

    Parameters
    ----------
    session_list : list
        List of lists of sessions to be concatenated.
    triggers : list
        List of triggers to be used.

    Returns
    -------
    Xs : np.array
        Data array.
    ys : np.array
        Label array
    sesh_inds : np.array
        Array of session indices.
    """
    
    for i, sesh in enumerate(session_list):
        sesh = [f'{i}-epo.fif' for i in sesh]
        X, y = read_and_concate_sessions(sesh, triggers)

        # replace all values of y with the trial number
        y = np.array([i for i in range(len(y))])

        # create an array of session indices
        sesh_ind = np.array([i for _ in range(len(y))])

        if i == 0:
            Xs = X
            ys = y
            sesh_inds = sesh_ind
        else:
            Xs = np.concatenate((Xs, X), axis=1)
            ys = np.concatenate((ys, y), axis=0)
            sesh_inds = np.concatenate((sesh_inds, sesh_ind), axis=0)

        
    return Xs, ys, sesh_inds


def get_triggers(trial_type:str = "animate"):
    path = Path(__file__)

    with open(path.parents[2] / 'info_files' / 'event_ids.txt', 'r') as f:
        file = f.read()
        event_ids = json.loads(file)

    triggers = [value for key, value in event_ids.items() if trial_type.capitalize() in key][:27]

    return triggers

if __name__ == '__main__':
    args = parse_args()

    # defining paths
    path = Path(__file__)
    output_path_pred = path.parent / "results" / f'{args.trial_type}_{args.task}_predict_trial_number.npy'
    output_path_true = path.parent / "results" / f'{args.trial_type}_{args.task}_true_trial_number.npy'
    logger_path = path.parent / 'logs' / f'{args.trial_type}_{args.task}_predict_trial_number.log'
    # ensure that the results directory exists
    output_path_pred.parents[0].mkdir(parents=True, exist_ok=True)
    
    sessions = [['visual_03', 'visual_04'], ['visual_05', 'visual_06', 'visual_07'], ['visual_08', 'visual_09', 'visual_10'], ['visual_11', 'visual_12', 'visual_13'], ['memory_01', 'memory_02'], ['memory_03', 'memory_04', 'memory_05', 'memory_06'],  ['memory_07', 'memory_08', 'memory_09', 'memory_10', 'memory_11'], ['memory_12', 'memory_13', 'memory_14', 'memory_15'] ,['visual_14', 'visual_15', 'visual_16', 'visual_17', 'visual_18', 'visual_19'],['visual_23', 'visual_24', 'visual_25', 'visual_26', 'visual_27', 'visual_28', 'visual_29'],['visual_30', 'visual_31', 'visual_32', 'visual_33', 'visual_34', 'visual_35', 'visual_36', 'visual_37', 'visual_38']]
    #  change sessions depending on the task
    if args.task == 'visual':
        sessions = sessions[:4] + sessions[8:]
    elif args.task == 'memory':
        sessions = sessions[4:8]
    # get the triggers
    triggers = get_triggers(args.trial_type)


    # prepare the data
    X, y, sesh_inds = prepare_data(sessions, triggers)

    # in this case the session number is the stratification variable
    pred, true = tgm_ridge_scores(X, y, stratify=sesh_inds, ncv=args.ncv)

    np.save(output_path_pred, pred)
    np.save(output_path_true, true)
"""
This script is decodes the session number using cross decoding or the bin number using within session decoding. 
This analysis can be run for both animate and inanimate trials, and for both the combined, visual and memory tasks.

"""

from pathlib import Path

import argparse as ap
import numpy as np
import json

# local imports
import sys
sys.path.append(str(Path(__file__).parents[2])) # adds the parent directory to the path so that the utils module can be imported
from utils.data.concatenate import read_and_concate_sessions
from utils.data.triggers import get_triggers_equal, balance_class_weights
from utils.data.prep_data import equalise_trials


def parse_args():
    parser = ap.ArgumentParser()
    
    # arg for indicating whether to decode time elapsed within session or between sessions
    # NOTE: WHEN TESTING AT CFIN MAKE SURE THIS DOES NOT RESULT IN A BUG
    parser.add_argument('--within', action='store_true', help='Indicates whether to decode time elapsed within session or between sessions.')
    parser.add_argument('--trial_type', type=str, default='animate', help='Trial type. Can be either animate or inanimate.')
    parser.add_argument('--task', type=str, default='combined', help='Task. Can be either combined, visual or memory.')
    args = parser.parse_args()
    
    return args


def prepare_within_data(session_list:list, triggers:list, n_splits:int=11):
    """
    Prepares data for decoding by concatenating sessions from the same day and splitting into n_splits. The y array is the bin number.

    Parameters
    ----------
    session_list : list
        List of sessions to be concatenated.
    triggers : list
        List of triggers to be used.
    n_splits : int, optional
        Number of splits to be used, by default 11
    """
    sesh = [f'{i}-epo.fif' for i in session_list]
    X, y = read_and_concate_sessions(sesh, triggers)

    # create splits
    Xs = np.array_split(X, n_splits, axis=1)

    # create y array where the value is the number of the given bin
    y = []
    for i, x in enumerate(Xs):
        n_trials = x.shape[1]
        y.append([i] * n_trials)
    
    y = np.concatenate(y)

    # balance class weights
    X, y, _ = balance_class_weights(X, y)

    return X, y


def prepare_between_data(session_list:list, triggers:list):
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
    X : np.array
        Data array.
    y : np.array
        Label array
    """
    
    Xs, ys = [], []

    for i, sesh in enumerate(session_list):
        sesh = [f'{i}-epo.fif' for i in sesh]
        X, y = read_and_concate_sessions(sesh, triggers)

        Xs.append(X)
        ys.append([i] * X.shape[1])

    # balance class weights
    X, y, _ = balance_class_weights(Xs, ys)

    return X, y




def get_triggers(trial_type:str = "animate"):
    path = Path(__file__)

    with open(path.parents[2] / 'info_files' / 'event_ids.txt', 'r') as f:
        file = f.read()
        event_ids = json.loads(file)

    triggers = [value for key, value in event_ids.items() if trial_type.capitalize() in key]

    return triggers

if __name__ == '__main__':
    args = parse_args()

    # defining paths
    path = Path(__file__)
    output_path = path / "accuracies" / f'{args.trial_type}_{args.task}_within.npy' if args.within else f'{args.trial_type}_{args.task}_between.npy'

    sessions = [['visual_03', 'visual_04'], ['visual_05', 'visual_06', 'visual_07'], ['visual_08', 'visual_09', 'visual_10'], ['visual_11', 'visual_12', 'visual_13'],['visual_14', 'visual_15', 'visual_16', 'visual_17', 'visual_18', 'visual_19'],['visual_23', 'visual_24', 'visual_25', 'visual_26', 'visual_27', 'visual_28', 'visual_29'],['visual_30', 'visual_31', 'visual_32', 'visual_33', 'visual_34', 'visual_35', 'visual_36', 'visual_37', 'visual_38'], ['memory_01', 'memory_02'], ['memory_03', 'memory_04', 'memory_05', 'memory_06'],  ['memory_07', 'memory_08', 'memory_09', 'memory_10', 'memory_11'], ['memory_12', 'memory_13', 'memory_14', 'memory_15']]
    
    # get triggers for the given trial type
    triggers = get_triggers(args.trial_type)


    if args.within:
        accuracies = np.zeros((len(sessions), 2)) # 2 for the two tasks

        # loop over sessions
        for i, session in enumerate(sessions):
            # load data
            Xs, ys = prepare_within_data(session, triggers)


    elif not args.within:
        # between session decoding

        # remember to use subsets of sessions for the different tasks
        pass
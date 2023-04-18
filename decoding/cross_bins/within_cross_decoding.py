"""
Mimics the cross-decoding analysis but within each session. 
Splits each session up into 11 parts, and trains and tests across all pairs of parts.


This is done in sensor space. 
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
import numpy as np
import multiprocessing as mp
import argparse
from functools import partial
from pathlib import Path

# local imports
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2])) # adds the parent directory to the path so that the utils module can be imported

from utils.data.concatenate import read_and_concate_sessions
from utils.data.triggers import get_triggers_equal, convert_triggers_animate_inanimate, balance_class_weights
from utils.analysis.decoder import Decoder
from cross_decoding import get_accuracy, equalise_trials

def parse_args():
    parser = argparse.ArgumentParser(description='This script is used to decode both source and sensor space data using cross decoding.')
    parser.add_argument('--model_type', type=str, default='LDA', help='Model type. Can be either LDA or RidgeClassifier.')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of cores to use.')
    parser.add_argument('--ncv', type=int, default=10, help='Number of cross validation folds.')
    parser.add_argument('--alpha', type=str, default='auto', help='Regularization parameter. Can be either auto or float.')

    args = parser.parse_args()

    assert args.n_jobs <= mp.cpu_count(), f'Number of jobs ({args.n_jobs}) cannot be larger than number of available cores ({mp.cpu_count()})'

    return args

def prep_data_split(session_list:list, triggers:list, n_splits:int=11):
    """
    Prepares data for decoding by concatenating sessions from the same day and splitting into n_splits.

    Parameters
    ----------
    session_list : list
        List of sessions to be concatenated.
    triggers : list
        List of triggers to be used.
    n_splits : int, optional
        Number of splits to be used, by default 11

    Returns
    -------
    Xs : list
        List of X arrays for each split.
    ys : list
        List of y arrays for each split.
    """

    sesh = [f'{i}-epo.fif' for i in session_list]
    X, y = read_and_concate_sessions(sesh, triggers)
    
    # converting triggers to animate and inanimate instead of images
    y = convert_triggers_animate_inanimate(y) 

    # create splits
    Xs = np.array_split(X, n_splits, axis=1)
    ys = np.array_split(y, n_splits)

    # balance class weights
    for i, (X, y) in enumerate(zip(Xs, ys)):
        Xs[i], ys[i], _ = balance_class_weights(X, y)

        # save the minumum number of trials
        if i == 0:
            min_trials = Xs[i].shape[1]
        else:
            if Xs[i].shape[1] < min_trials:
                min_trials = Xs[i].shape[1]

    # equalize number of trials in each split to the minimum number of trials
    Xs, ys = equalise_trials(Xs, ys)
    
    return Xs, ys

def get_accuracy_session(Xs:list, ys:list, decoder:Decoder, n_jobs:int=1):
    """
    Runs decoding analysis on one session.

    Parameters
    ----------
    Xs : list
        List of X arrays from one session with all splits.
    ys : list
        List of y arrays from one session with all splits.
    n_jobs : int
        Number of cores to use. Default is 1.
    decoder : Decoder
        Decoder object.

    Returns
    -------
    accuracies : np.ndarray
        Array of accuracies for each pair of splits.
    """

    # empty array to store accuracies
    accuracies = np.zeros((len(Xs), len(Xs), 250, 250))

    # loop over all pairs of sessions
    decoding_inputs = [(train_sesh, test_sesh, idx*i+i) for idx, train_sesh in enumerate(range(len(Xs))) for i, test_sesh in enumerate(range(len(Xs)))]

    # using partial to pass arguments to function that are not changing
    multi_parse = partial(get_accuracy, Xs, ys, decoder)

    # multiprocessing
    with mp.Pool(n_jobs) as p:
        for train_session, test_session, accuracy in p.map(multi_parse, decoding_inputs):
            accuracies[train_session, test_session, :, :] = accuracy
        
    p.close()
    p.join()

    return accuracies


def main():
    args = parse_args()

    path = Path(__file__)

    sessions = [['visual_03', 'visual_04'], ['visual_05', 'visual_06', 'visual_07'], ['visual_08', 'visual_09', 'visual_10'], ['visual_11', 'visual_12', 'visual_13'],['visual_14', 'visual_15', 'visual_16', 'visual_17', 'visual_18', 'visual_19'],['visual_23', 'visual_24', 'visual_25', 'visual_26', 'visual_27', 'visual_28', 'visual_29'],['visual_30', 'visual_31', 'visual_32', 'visual_33', 'visual_34', 'visual_35', 'visual_36', 'visual_37', 'visual_38'], ['memory_01', 'memory_02'], ['memory_03', 'memory_04', 'memory_05', 'memory_06'],  ['memory_07', 'memory_08', 'memory_09', 'memory_10', 'memory_11'], ['memory_12', 'memory_13', 'memory_14', 'memory_15']]
    
    # get triggers for equal number of trials per condition (27 animate and 27 inanimate)
    triggers = get_triggers_equal()

    # preparing the data for decoding
    for i, session_list in enumerate(sessions):
        if i == 0:
            Xs, ys = prep_data_split(session_list, triggers, n_splits=11) 
        else:
            Xs_tmp, ys_tmp = prep_data_split(session_list, triggers, n_splits=11)
            
            # loop over splits
            for j, (X, y) in enumerate(zip(Xs_tmp, ys_tmp)):
                Xs[j] = np.concatenate((Xs[j], X), axis=1)
                ys[j] = np.concatenate((ys[j], y), axis=0)

    # Make sure all splits have the same number of trials
    Xs, ys = equalise_trials(Xs, ys)

    # preparing the decoder
    decoder = Decoder(classification=True, ncv=args.ncv, alpha=args.alpha, model_type=args.model_type, get_tgm = True, verbose=False)

    # get accuracies
    accuracies = get_accuracy_session(Xs, ys, decoder, n_jobs=args.n_jobs)

    # save accuracies
    out_path = path.parent / "accuracies_within" / f"{args.model_type}_{args.alpha}_{args.ncv}.npy"
        
    # ensure accuracy directory exists
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True)
            
    np.save(out_path, accuracies)


if __name__ == '__main__':
    main()



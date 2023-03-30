"""
This script is used to decode both source and sensor space data using cross decoding.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
import numpy as np
import os
import multiprocessing as mp
from time import perf_counter
import argparse
from functools import partial

# local imports
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2])) # adds the parent directory to the path so that the utils module can be imported

from utils.data.concatenate import flip_sign, read_and_concate_sessions_source, read_and_concate_sessions
from utils.data.triggers import get_triggers_equal, convert_triggers_animate_inanimate, balance_class_weights
from utils.analysis.decoder import Decoder


""" -------- DEFINE FUNCTIONS -------- """

def parse_args():
    parser = argparse.ArgumentParser(description='This script is used to decode both source and sensor space data using cross decoding.')
    parser.add_argument('--model_type', type=str, default='LDA', help='Model type. Can be either LDA or RidgeClassifier.')
    parser.add_argument('--parc', type=str, default='sens', help='Parcellation. Can be either aparc, aparc.a2009s, aparc.DKTatlas, HCPMMP1 or sens.')
    parser.add_argument('--n_jobs', type=int, default=10, help='Number of cores to use.')
    parser.add_argument('--ncv', type=int, default=10, help='Number of cross validation folds.')
    parser.add_argument('--alpha', type=str, default='auto', help='Regularization parameter. Can be either auto or float.')

    args = parser.parse_args()

    assert args.n_jobs <= mp.cpu_count(), f'Number of jobs ({args.n_jobs}) cannot be larger than number of available cores ({mp.cpu_count()})'

    return args


def prep_data(sessions, triggers, parc, path, event_path):
    Xs, ys = [], []

    for sesh in sessions:
        if parc == 'sens':
            sesh = [f'{i}-epo.fif' for i in sesh] # WITH ICA - should it be without?
            X, y = read_and_concate_sessions(sesh, triggers)
        else:
            X, y = read_and_concate_sessions_source(path, event_path, sesh, triggers)
        
        y = convert_triggers_animate_inanimate(y) # converting triggers to animate and inanimate instead of images
        
        # balance class weights
        X, y, _ = balance_class_weights(X, y)

        Xs.append(X)
        ys.append(y)
    
    return Xs, ys

def get_accuracy(input:tuple, Xs, ys, classification:bool=True, ncv:int=10, alpha:str='auto', model_type:str='LDA'):
    """
    This function is used to decode both source and sensor space data using cross decoding.

    Args:
        input (tuple): tuple containing session_train, session_test and idx
        classification (bool, optional): whether to perform classification or regression. Defaults to classification.
        ncv (int, optional): number of cross validation folds. Defaults to ncv.
    
    Returns:
        tuple: tuple containing session_train, session_test and accuracy
    """
    start = perf_counter()
    (session_train, session_test, idx, ncv, alpha) = input # unpacking input tuple

    decoder = Decoder(classification=classification, ncv = ncv, alpha = alpha, model_type = model_type, get_tgm=True)

    if session_test == session_train: # avoiding double dipping within session, by using within session decoder
        X = Xs[session_train]
        y = ys[session_train]
        accuracy = decoder.run_decoding(X, y)

    else:
        X_train = Xs[session_train]
        X_test = Xs[session_test]

        y_train = ys[session_train]
        y_test = ys[session_test]

        accuracy = decoder.run_decoding_across_sessions(X_train, y_train, X_test, y_test)
    
    end = perf_counter()

    print(f'Finished decoding index {idx} in {round(end-start, 2)} seconds')

    return session_train, session_test, accuracy

def equal_trials(X, y, n: int):
    """
    Removes trials from X and y, such that the number of trials is equal to n. It is assumed that classes are already balanced. Therefore an equal number of trials is removed from each class.

    Args:
        X (np.array): data
        y (np.array): labels (0 or 1)
        n (int): number of trials to keep
    
    Returns:
        tuple: tuple containing X and y with n trials
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
    X = np.delete(X, idx, axis=1)
    y = np.delete(y, idx)

    return X, y


def main():

    """ -------- PARSE ARGUMENTS -------- """

    args = parse_args()

    classification = True
    model_type = args.model_type
    get_tgm = True
    parc = args.parc
    n_jobs = args.n_jobs
    ncv = args.ncv
    alpha = args.alpha

    output_path = os.path.join('accuracies', f'cross_decoding_10_{model_type}_{parc}.npy')


    # define logger
    logger = logging.getLogger(__name__)

    # log info
    logger.info(f'Running cross decoding with the following parameters...')
    logger.info(f'Classification: {classification}')
    logger.info(f'Number of cross validation folds: 10')
    logger.info(f'Parcellation: {parc}')

    sessions = [['visual_03', 'visual_04'], ['visual_05', 'visual_06', 'visual_07'], ['visual_08', 'visual_09', 'visual_10'], ['visual_11', 'visual_12', 'visual_13'],['visual_14', 'visual_15', 'visual_16', 'visual_17', 'visual_18', 'visual_19'],['visual_23', 'visual_24', 'visual_25', 'visual_26', 'visual_27', 'visual_28', 'visual_29'],['visual_30', 'visual_31', 'visual_32', 'visual_33', 'visual_34', 'visual_35', 'visual_36', 'visual_37', 'visual_38'], ['memory_01', 'memory_02'], ['memory_03', 'memory_04', 'memory_05', 'memory_06'],  ['memory_07', 'memory_08', 'memory_09', 'memory_10', 'memory_11'], ['memory_12', 'memory_13', 'memory_14', 'memory_15']]

    triggers = get_triggers_equal() # get triggers for equal number of trials per condition (27 animate and 27 inanimate)

    path = os.path.join(os.path.sep, 'media', '8.1', 'final_data', 'laurap', 'source_space', 'parcelled', parc)
    event_path = os.path.join('..', '..', 'info_files', 'events')
    
    # reading and concatenating data
    Xs, ys = prep_data(sessions, triggers, parc, path, event_path)
    
    # sign flipping for concatenated data (source space only)
    if parc != 'sens': 
        Xs = [flip_sign(Xs[0], X) for X in Xs]

    n_trials = [X.shape[1] for X in Xs]
    
    # max number of trials
    min_trials = min(n_trials)

    # make sure all sessions have the same number of trials
    for i,(X, y) in enumerate(zip(Xs, ys)):
        Xs[i], ys[i] = equal_trials(X, y, min_trials)

    # preparing decoding inputs for multiprocessing
    decoding_inputs = [(train_sesh, test_sesh, idx*i+i, ncv, alpha) for idx, train_sesh in enumerate(range(len(Xs))) for i, test_sesh in enumerate(range(len(Xs)))]

    # empty array to store accuracies in
    accuracies = np.zeros((len(Xs), len(Xs), Xs[0].shape[0], Xs[0].shape[0]), dtype=float)

    # using partial to pass arguments to function that are not changing
    multi_parse = partial(get_accuracy, Xs, ys, classification=classification, model_type=model_type, get_tgm=get_tgm)

    with mp.Pool(n_jobs) as p:
        for train_session, test_session, accuracy in p.map(multi_parse, decoding_inputs):
            accuracies[train_session, test_session, :, :] = accuracy
    
    p.close()
    p.join()

    # saving accuracies to file
    if not os.path.exists('accuracies'):
        os.mkdir('accuracies')

    np.save(output_path, accuracies)
    

if __name__ == '__main__':
    main()

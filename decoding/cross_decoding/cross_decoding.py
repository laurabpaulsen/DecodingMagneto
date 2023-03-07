"""
This script is used to decode both source and sensor space data using cross decoding.

Dev notes:
- [ ] Check that decoding in parcellated space works as well as sensor space
    - [X] aparc
    - [X] aparc.a2009s
    - [X] aparc.DKTatlas
    - [X] sens
    - [ ] HCPMMP1
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
import numpy as np
import os
import multiprocessing as mp
from time import perf_counter

# local imports
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2])) # adds the parent directory to the path so that the utils module can be imported

from utils.data.concatenate import flip_sign, read_and_concate_sessions_source, read_and_concate_sessions
from utils.data.triggers import get_triggers_equal, convert_triggers_animate_inanimate
from utils.analysis.decoder import Decoder

""" -------- SET PARAMETERS -------- """
classification = True
ncv = 10
alpha = 'auto'
model_type = 'LDA' # can be either LDA, SVM or RidgeClassifier
get_tgm = True
parc = 'sens' # can be either aparc, aparc.a2009s, aparc.DKTatlas or sens
ncores = 12 # mp.cpu_count()
output_path = os.path.join('accuracies', f'cross_decoding_{ncv}_{model_type}_{parc}.npy')

""" -------- DEFINE FUNCTIONS -------- """
def prep_data(sessions, triggers, parc, path, event_path):
    Xs, ys = [], []

    for sesh in sessions:
        if parc == 'sens':
            sesh = [f'{i}-epo.fif' for i in sesh] # WITH ICA - should it be without?
            X, y = read_and_concate_sessions(sesh, triggers)
        else:
            X, y = read_and_concate_sessions_source(path, event_path, sesh, triggers)
        
        y = convert_triggers_animate_inanimate(y) # converting triggers to animate and inanimate instead of images

        Xs.append(X)
        ys.append(y)
    
    return Xs, ys

def get_accuracy(input:tuple, classification=classification, ncv=ncv):
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

    decoder = Decoder(classification=classification, ncv = ncv, alpha = alpha, scale = True, model_type = model_type, get_tgm=True)
    
    (session_train, session_test, idx) = input # unpacking input tuple

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

    print(f'Finished decoding index {idx} in {end-start} seconds')

    return session_train, session_test, accuracy

if __name__ == '__main__':
    # define logger
    logger = logging.getLogger(__name__)

    # log info
    logger.info(f'Running cross decoding with the following parameters...')
    logger.info(f'Classification: {classification}')
    logger.info(f'Number of cross validation folds: {ncv}')
    logger.info(f'Parcellation: {parc}')
    logger.info(f'Alpha: {alpha}')


    sessions = [['visual_03', 'visual_04'], ['visual_05', 'visual_06', 'visual_07'], ['visual_08', 'visual_09', 'visual_10'], ['visual_11', 'visual_12', 'visual_13'],['visual_14', 'visual_15', 'visual_16', 'visual_17', 'visual_18', 'visual_19'],['visual_23', 'visual_24', 'visual_25', 'visual_26', 'visual_27', 'visual_28', 'visual_29'],['visual_30', 'visual_31', 'visual_32', 'visual_33', 'visual_34', 'visual_35', 'visual_36', 'visual_37', 'visual_38'], ['memory_01', 'memory_02'], ['memory_03', 'memory_04', 'memory_05', 'memory_06'],  ['memory_07', 'memory_08', 'memory_09', 'memory_10', 'memory_11'], ['memory_12', 'memory_13', 'memory_14', 'memory_15']]

    triggers = get_triggers_equal() # get triggers for equal number of trials per condition (27 animate and 27 inanimate)

    path = os.path.join(os.path.sep, 'media', '8.1', 'final_data', 'laurap', 'source_space', 'parcelled', parc)
    event_path = os.path.join('..', '..', 'info_files', 'events')
    
    # reading and concatenating data
    Xs, ys = prep_data(sessions, triggers, parc, path, event_path)
    
    # sign flipping for concatenated data (source space only)
    if parc != 'sens': 
        Xs = [flip_sign(Xs[0], X) for X in Xs]

    # preparing decoding inputs for multiprocessing
    decoding_inputs = [(train_sesh, test_sesh, idx) for idx, train_sesh in enumerate(range(len(Xs))) for test_sesh in range(len(Xs))]

    # empty array to store accuracies in
    accuracies = np.zeros((len(Xs), len(Xs), Xs[0].shape[0], Xs[0].shape[0]), dtype=float)

    with mp.Pool(ncores) as p:
        for train_session, test_session, accuracy in p.map(get_accuracy, decoding_inputs):
            accuracies[train_session, test_session, :, :] = accuracy
    
    p.close()
    p.join()

    # saving accuracies to file
    if not os.path.exists('accuracies'):
        os.mkdir('accuracies')

    np.save(output_path, accuracies)
    

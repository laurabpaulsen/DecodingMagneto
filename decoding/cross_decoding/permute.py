"""
This script is used conduct permutation testing on the cross decoding accuracies (only diagonal of the TGM) to test for significance.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
import numpy as np
import multiprocessing as mp
from time import perf_counter
import argparse
from functools import partial
from pathlib import Path

# local imports
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2])) # adds the parent directory to the path so that the utils module can be imported

from utils.data.concatenate import read_and_concate_sessions
from utils.data.triggers import get_triggers_equal, convert_triggers_animate_inanimate, balance_class_weights
from utils.data.prep_data import equalise_trials
from utils.analysis.decoder import Decoder


""" -------- DEFINE FUNCTIONS -------- """

def prep_data(sessions, triggers):
    Xs, ys = [], []

    for sesh in sessions:
        sesh = [f'{i}-epo.fif' for i in sesh] 
        X, y = read_and_concate_sessions(sesh, triggers)
        
        # converting triggers to animate and inanimate instead of images
        y = convert_triggers_animate_inanimate(y) 
        
        # balance class weights
        X, y, _ = balance_class_weights(X, y)

        Xs.append(X)
        ys.append(y)
    
    return Xs, ys

def get_accuracy(Xs:list, ys:list, decoder:Decoder, input:tuple):
    """
    This function is used to decode both source and sensor space data using cross decoding.

    Parameters
    ----------
    Xs : list
        list of X arrays
    ys : list
        list of y arrays
    decoder : Decoder
        Decoder object
    input : tuple
        tuple containing ind_train, ind_test and idx
    
    Returns
    -------
    (ind_train, ind_test, accuracy) : tuple
        tuple containing ind_train, ind_test and accuracy
    """
    start = perf_counter()

    (ind_train, ind_test, idx) = input # unpacking input tuple

    if ind_test == ind_train: # avoiding double dipping within session, by using within session decoder
        accuracy = decoder.run_decoding(Xs[ind_train], ys[ind_train])

    else: # using cross decoding
        accuracy = decoder.run_decoding_across_sessions(Xs[ind_train], ys[ind_train], Xs[ind_test], ys[ind_test])
    
    end = perf_counter()

    print(f'Finished decoding index {idx} in {round(end-start, 2)} seconds')

    return ind_train, ind_test, accuracy


def main():
    parc = "sens"
    ncv = 10
    n_permutations = 100
    n_jobs = mp.cpu_count() - 4

    # set random seed
    np.random.seed(42)

    # defining paths
    path = Path(__file__).parent
    output_path = path / "accuracies" 

    # define logger
    logger = logging.getLogger(__name__)

    # log info
    logger.info(f'Running cross decoding with the following parameters...')
    logger.info(f'Number of cross validation folds: {ncv}')
    logger.info(f'Parcellation: {parc}')
    logger.info(f'Number of permutations: {n_permutations}')
    logger.info(f'Number of jobs: {n_jobs}')

    sessions = [['visual_03', 'visual_04'], ['visual_05', 'visual_06', 'visual_07'], ['visual_08', 'visual_09', 'visual_10'], ['visual_11', 'visual_12', 'visual_13'],['visual_14', 'visual_15', 'visual_16', 'visual_17', 'visual_18', 'visual_19'],['visual_23', 'visual_24', 'visual_25', 'visual_26', 'visual_27', 'visual_28', 'visual_29'],['visual_30', 'visual_31', 'visual_32', 'visual_33', 'visual_34', 'visual_35', 'visual_36', 'visual_37', 'visual_38'], ['memory_01', 'memory_02'], ['memory_03', 'memory_04', 'memory_05', 'memory_06'],  ['memory_07', 'memory_08', 'memory_09', 'memory_10', 'memory_11'], ['memory_12', 'memory_13', 'memory_14', 'memory_15']]

    triggers = get_triggers_equal() # get triggers for equal number of trials per condition (27 animate and 27 inanimate)
    
    # reading and concatenating data
    Xs, ys = prep_data(sessions, triggers)

    # equalising number of trials
    Xs, ys = equalise_trials(Xs, ys)
    
    # empty array to store accuracies in
    accuracies = np.zeros((len(Xs), len(Xs), Xs[0].shape[0], n_permutations), dtype=float)

    # preparing the decoder
    decoder = Decoder(classification=True, ncv = ncv, alpha = "auto", model_type = "LDA", get_tgm=False)

    for perm in range(n_permutations):
        ys = [np.random.permutation(y) for y in ys]

        # preparing decoding inputs for multiprocessing
        decoding_inputs = [(train_sesh, test_sesh, idx*i+i) for idx, train_sesh in enumerate(range(len(Xs))) for i, test_sesh in enumerate(range(len(Xs)))]


        # using partial to pass arguments to function that are not changing
        multi_parse = partial(get_accuracy, Xs, ys, decoder)

        with mp.Pool(n_jobs) as p:
            for train_session, test_session, accuracy in p.map(multi_parse, decoding_inputs):
                accuracies[train_session, test_session, :, perm] = accuracy
        
        p.close()
        p.join()

    # making sure output path exists
    if not output_path.exists():
        output_path.mkdir(parents=True)
   
    # saving accuracies
    np.save(output_path / f'permutations_{ncv}_LDA_{parc}.npy', accuracies)
    

if __name__ == '__main__':
    main()

"""
dev notes:
- [ ] do we want to add parallel processing for this script??? (maybe not necessary)

"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2]))
sys.path.append('../cross_decoding')

import argparse
import numpy as np
import os
import multiprocessing
import logging


from utils.analysis.decoder import Decoder
from utils.data.concatenate import read_and_concate_sessions
from utils.data.triggers import get_triggers_equal, convert_triggers_animate_inanimate, balance_class_weights
from cross_decoding import equal_trials



def parse_args():
    parser = argparse.ArgumentParser(description='Decoding analysis')
    parser.add_argument('--ica', type=str, default='ica', help='ica or no_ica')
    parser.add_argument('--n_jobs', type=int, default=1, help='number of jobs')
    parser.add_argument('--model_type', type=str, default='LDA', help='model type')
    parser.add_argument('--ncv', type=int, default=10, help='number of cross validations')
    parser.add_argument('--alpha', type=str, default='auto', help='alpha level')
    parser.add_argument('--classification', type=bool, default=True, help='classification or regression')
    parser.add_argument('--tgm', type=bool, default=True, help='tgm or not')
    args = parser.parse_args()

    # check if ica or no_ica
    if args.ica not in ['ica', 'no_ica']:
        raise ValueError('Input can only be ica or no_ica')
    
    return args

def decode_session(list_of_sessions, session_number, ica):
    """
    Within session decoding on a single session. Reads in the data, converts the triggers, balances the class weights, runs decoding and saves the results.
    
    Parameters
    ----------
    list_of_sessions : list
        list of sessions within a single day
    session_number : int
        the session number
    ica : str
        ica or no_ica
    
    Returns
    -------
    None
    """
    triggers = get_triggers_equal()

    X, y = read_and_concate_sessions(list_of_sessions, triggers)

    # animate vs inanimate triggers instead of image triggers
    y = convert_triggers_animate_inanimate(y)

    # balance class weights
    X, y, _= balance_class_weights(X, y)
        
    # equalize number of trials (so all sessions have the same number of trials)
    X, y = equal_trials(X, y, 588)

    # run decoding
    accuracies = decoder.run_decoding(X, y)
        
    # save results
    np.save(f'accuracies/{ica}_session_{session_number+1}.npy', accuracies)


def main():
    args = parse_args()

    # initialize logger
    logging.basicConfig(filename='ica_decoding.log', level=logging.DEBUG)

    logging.info('Starting decoding analysis with the following parameters:')
    logging.info(f'ica: {args.ica}')
    logging.info(f'n_jobs: {args.n_jobs}')
    logging.info(f'model_type: {args.model_type}')
    logging.info(f'ncv: {args.ncv}')
    logging.info(f'alpha: {args.alpha}')
    logging.info(f'classification: {args.classification}')
    logging.info(f'tgm: {args.tgm}')
    

    # path to the data
    path = os.path.join(os.pathsep, 'media', '8.1', 'final_data', 'laurap', 'epochs')
    sessions = [['visual_03', 'visual_04'], ['visual_05', 'visual_06', 'visual_07'], ['visual_08', 'visual_09', 'visual_10'], ['visual_11', 'visual_12', 'visual_13'],['visual_14', 'visual_15', 'visual_16', 'visual_17', 'visual_18', 'visual_19'],['visual_23', 'visual_24', 'visual_25', 'visual_26', 'visual_27', 'visual_28', 'visual_29'],['visual_30', 'visual_31', 'visual_32', 'visual_33', 'visual_34', 'visual_35', 'visual_36', 'visual_37', 'visual_38'], ['memory_01', 'memory_02'], ['memory_03', 'memory_04', 'memory_05', 'memory_06'],  ['memory_07', 'memory_08', 'memory_09', 'memory_10', 'memory_11'], ['memory_12', 'memory_13', 'memory_14', 'memory_15']]
    
    decoder = Decoder(classification=args.classification, ncv = args.ncv, alpha = args.alpha, scale = True, model_type = args.model_type, get_tgm=args.get_tgm)
    
    # check if accuracies folder exists, if not create it
    if not os.path.exists('accuracies'):
        os.makedirs('accuracies')

    # loop over sessions one pool for each session
    if args.ica == 'ica':
        sesh = [[f'{s}-epo.fif' for s in session] for session in sessions]
    elif args.ica == 'no_ica':
        sesh = [[f'{s}-no_ica-epo.fif' for s in session] for session in sessions]

    pool = multiprocessing.Pool(processes = args.n_jobs)

    for session_number, session in enumerate(sesh):
        pool.apply_async(decode_session, args=(session, session_number, args.ica))
    
    pool.close()
    pool.join()

if __name__ in '__main__':
    main()
   
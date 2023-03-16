"""
dev notes:
- [ ] do we want to add parallel processing for this script??? (maybe not necessary)

"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2]))
sys.path.append('../cross_decoding')

import argparse
from tqdm import tqdm

from utils.analysis.decoder import Decoder
from utils.data.concatenate import read_and_concate_sessions
from utils.data.triggers import get_triggers_equal, convert_triggers_animate_inanimate, balance_class_weights

from cross_decoding import equal_trials

import numpy as np
import os

classification = True
ncv = 10
alpha = "auto"
model_type = 'LDA'
get_tgm = True

def parse_args():
    parser = argparse.ArgumentParser(description='Decoding analysis')
    parser.add_argument('--ica', type=str, default='ica', help='ica or no_ica')
    args = parser.parse_args()

    # check if ica or no_ica
    if args.ica not in ['ica', 'no_ica']:
        raise ValueError('Input can only be ica or no_ica')
    
    return args

if __name__ in '__main__':
    args = parse_args()

    # path to the data
    path = os.path.join(os.pathsep, 'media', '8.1', 'final_data', 'laurap', 'epochs')
    sessions = [['visual_03', 'visual_04'], ['visual_05', 'visual_06', 'visual_07'], ['visual_08', 'visual_09', 'visual_10'], ['visual_11', 'visual_12', 'visual_13'],['visual_14', 'visual_15', 'visual_16', 'visual_17', 'visual_18', 'visual_19'],['visual_23', 'visual_24', 'visual_25', 'visual_26', 'visual_27', 'visual_28', 'visual_29'],['visual_30', 'visual_31', 'visual_32', 'visual_33', 'visual_34', 'visual_35', 'visual_36', 'visual_37', 'visual_38'], ['memory_01', 'memory_02'], ['memory_03', 'memory_04', 'memory_05', 'memory_06'],  ['memory_07', 'memory_08', 'memory_09', 'memory_10', 'memory_11'], ['memory_12', 'memory_13', 'memory_14', 'memory_15']]
    
    decoder = Decoder(classification=classification, ncv = ncv, alpha = alpha, scale = True, model_type = model_type, get_tgm=get_tgm)
    triggers = get_triggers_equal()

    # check if accuracies folder exists, if not create it
    if not os.path.exists('accuracies'):
        os.makedirs('accuracies')
   
    for i, session in tqdm(enumerate(sessions)):
        # load data
        if args.ica == 'ica':
            sesh = [f'{s}-epo.fif' for s in session]
        elif args.ica == 'no_ica':
            sesh = [f'{s}-no_ica-epo.fif' for s in session]

        X, y = read_and_concate_sessions(sesh, triggers)

        # balance class weights
        X, y, _= balance_class_weights(X, y)

        print(f'X shape: {X.shape}', f'y shape: {y.shape}')
        # animate vs inanimate triggers instead of image triggers
        y = convert_triggers_animate_inanimate(y)

        # equalize number of trials (so all sessions have the same number of trials)
        X, y = equal_trials(X, y, 588)

        # run decoding
        accuracies = decoder.run_decoding(X, y)
        
        # save results
        np.save(f'accuracies/accuracies_{args.ica}_session_{i+1}.npy', accuracies)
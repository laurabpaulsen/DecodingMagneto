"""
Notes: Do we want to do this in source or sensor space?
"""
import os
import numpy as np
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parents[2])) # adds the parent directory to the path so that the utils module can be imported


from utils.analysis.decoder import Decoder
from utils.data.triggers import balance_class_weights
from utils.data.concatenate import flip_sign
import pickle as pkl
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def read_and_concat(file_list):
    for f in tqdm(file_list):
        # get session number and type
        session = f.split('/')[-1]
        session = session.split('_parcelled')[0]

        # load y 
        with open(os.path.join('..', '..', 'info_files', 'events', f'{session}_events.pkl'), 'rb') as y_f:
            y_tmp = pkl.load(y_f)

        X_tmp = np.load(f)
        X_tmp = X_tmp.transpose(2, 0, 1)

        if file_list.index(f) == 0:
            X = X_tmp
            y = y_tmp
        else:
            # sign flip
            X_tmp = flip_sign(X, X_tmp)

            X = np.concatenate((X, X_tmp), axis=1)
            y = np.concatenate((y, y_tmp))

    return X, y

if __name__ in "__main__":
    # list all files in the aparc.a2009s directory
    path = os.path.join(os.path.sep, 'media', '8.1', 'final_data', 'laurap', 'source_space', 'parcelled', 'aparc.a2009s')
    files = os.listdir(path)

    memory_files = [os.path.join(path, f) for f in files if 'memory' in f]
    visual_files = [os.path.join(path, f)for f in files if 'visual' in f]

    # load data
    logging.info('Loading data...')
    X_memory, y_memory = read_and_concat(memory_files)
    X_visual, y_visual = read_and_concat(visual_files)

    # concatenate data
    X = np.concatenate((X_memory, X_visual), axis=1)
    y = [1] * len(y_memory) + [0] * len(y_visual)

    # shuffle indices
    indices = np.arange(X.shape[1])
    np.random.shuffle(indices)

    X = X[:, indices, :]
    y = y[indices]

    # ensure equal number of memory and visual trials
    logging.info('Balancing class weights...')
    X, y, _ = balance_class_weights(X, y)

    # run decoding
    logging.info('Running decoding...')
    decoder = Decoder(classification=True, ncv=10, alpha='auto', scale=True, model_type='LDA', get_tgm=True)
    accuracies = decoder.run_decoding(X, y)

    # check if directory exists if not create it
    if not os.path.exists('accuracies'):
        os.makedirs('accuracies')
    
    # save results
    np.save('accuracies/accuracies_memory_or_visual.npy', accuracies)

"""

"""
import os
import numpy as np
import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parents[2])) # adds the parent directory to the path so that the utils module can be imported

from utils.analysis.decoder import Decoder
from utils.data.triggers import balance_class_weights, get_triggers_equal
from utils.data.concatenate import read_and_concate_sessions

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

if __name__ in "__main__":
    path = os.path.join(os.path.sep, 'media', '8.1', 'final_data', 'laurap', 'epochs')
    files = os.listdir(path)
    
    # remove no_ica files
    files = [f for f in files if 'no_ica' not in f]

    memory_files = [f for f in files if 'memory' in f]
    visual_files = [f for f in files if 'visual' in f]

    # load data
    logging.info('Loading data...')
    X_memory, y_memory = read_and_concate_sessions(memory_files, get_triggers_equal())
    X_visual, y_visual = read_and_concate_sessions(visual_files, get_triggers_equal())

    # concatenate data
    X = np.concatenate((X_memory, X_visual), axis=1)
    y = np.array([1] * len(y_memory) + [0] * len(y_visual))

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
    decoder = Decoder(classification=True, ncv=10, alpha='auto', model_type='LDA', get_tgm=True)
    accuracies = decoder.run_decoding(X, y)

    # check if directory exists if not create it
    if not os.path.exists('accuracies'):
        os.makedirs('accuracies')
    
    # save results
    np.save('accuracies/accuracies_memory_or_visual.npy', accuracies)

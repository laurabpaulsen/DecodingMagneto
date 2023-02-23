"""
Notes: Do we want to do this in source or sensor space?


"""
import os
import numpy as np
from utils.analysis.decoder import Decoder
from utils.data_prep.concatenate import read_and_concate_sessions_source
from utils.data_prep.triggers import balance_class_weights


if __name__ in "__main__":
    # list all files in the aparc.a2009s directory
    files = os.listdir(os.path.join('aparc.a2009s'))

    memory_files = [file for file in files if 'memory' in file]
    visual_files = [file for file in files if 'visual' in file]

    # load data
    X_memory, y_memory = read_and_concate_sessions_source(memory_files)
    X_visual, y_visual = read_and_concate_sessions_source(visual_files)

    y_memory = [1 for _ in y_memory] # 1 = memory
    y_visual = [0 for _ in y_visual] # 0 = visual

    # concatenate data
    X = np.concatenate((X_memory, X_visual), axis=1)
    y = np.concatenate((y_memory, y_visual), axis=1)

    # shuffle indices
    indices = np.arange(X.shape[1])
    np.random.shuffle(indices)

    X = X[:, indices, :]
    y = y[indices]

    # ensure equal number of memory and visual trials
    X, y = balance_class_weights(X, y)

    # run decoding
    decoder = Decoder(classification=True, ncv=10, alpha='auto', scale=True, model_type='LDA', get_tgm=True)
    accuracies = decoder.run_decoding(X, y, decoder)

    # check if directory exists if not create it
    if not os.path.exists('accuracies'):
        os.makedirs('accuracies')
    
    # save results
    np.save('accuracies/accuracies_memory_or_visual.npy', accuracies)

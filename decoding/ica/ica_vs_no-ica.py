"""
dev notes:
- [ ] do we want to add parallel processing for this script??? (maybe not necessary)

"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2]))

from utils.analysis.decoder import Decoder
from utils.data_prep.concatenate import read_and_concate_sessions
from utils.data_prep.triggers import get_triggers_equal, convert_triggers_animate_inanimate

import numpy as np
import os

classification = True
ncv = 10
alpha = "auto"
model_type = 'LDA'
get_tgm = True



if __name__ in '__main__':

    triggers = get_triggers_equal()

    # path to the data
    path = '/media/8.1/final_data/laurap/epochs'
    sessions = [['visual_03', 'visual_04'], ['visual_05', 'visual_06', 'visual_07'], ['visual_08', 'visual_09', 'visual_10'], ['visual_11', 'visual_12', 'visual_13'],['visual_14', 'visual_15', 'visual_16', 'visual_17', 'visual_18', 'visual_19'],['visual_23', 'visual_24', 'visual_25', 'visual_26', 'visual_27', 'visual_28', 'visual_29'],['visual_30', 'visual_31', 'visual_32', 'visual_33', 'visual_34', 'visual_35', 'visual_36', 'visual_37', 'visual_38'], ['memory_01', 'memory_02'], ['memory_03', 'memory_04', 'memory_05', 'memory_06'],  ['memory_07', 'memory_08', 'memory_09', 'memory_10', 'memory_11'], ['memory_12', 'memory_13', 'memory_14', 'memory_15']]
    
    decoder = Decoder(classification=classification, ncv = ncv, alpha = alpha, scale = True, model_type = model_type, get_tgm=get_tgm)

    for i, session in enumerate(sessions):
        # load data
        sesh_ica = [f'{sesh}-epo.fif' for sesh in session]
        sesh_no_ica = [f'{sesh}-no_ica-epo.fif' for sesh in session]
        
        X, y = read_and_concate_sessions(sesh_ica, triggers)
        X_no_ica, y_no_ica = read_and_concate_sessions(sesh_no_ica, triggers)

        # animate vs inanimate triggers instead of image triggers
        y = convert_triggers_animate_inanimate(y)
        y_no_ica = convert_triggers_animate_inanimate(y_no_ica)

        # run decoding
        accuracies = decoder.run_decoding(X, y, decoder)
        accuracies_no_ica = decoder.run_decoding(X_no_ica, y_no_ica, decoder)


        # check if directory exists
        if not os.path.exists('accuracies'):
            os.makedirs('accuracies')
        
        # save results
        np.save(f'accuracies/accuracies_session_{i+1}.npy', accuracies)
        np.save(f'accuracies/accuracies_no_ica_session_{i+1}.npy', accuracies_no_ica)

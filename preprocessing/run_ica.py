'''
Usage, e.g., python run_ica.py -i '/media/8.1/raw_data/raw_data/memory_01.fif'

This script is used for initial preprocessing. The following steps are included:
    1) Excludes bad-channels based on file 'session_info.txt'. These channels were marked as bad based on visual expection of raw MEG data.
    2) Crops the ends of the MEG recordings according to times specified in 'session_info.txt'
    3) High and low pass filtering
    4) Running independent component analysis (ICA)

The script saves the ICA to a file. Unwanted components are manually detected and removed the file 'check_ica.ipynb'.
'''

import argparse
import mne
import json

def main(filepath):
    filename = filepath.split('/')[-1]
    outpath =  '/media/8.1/intermediate_data/laurap/ica/ica_solution/' + filename.split('.')[0] + '-ica.fif'
    
    # loading in the raw data
    raw = mne.io.read_raw_fif(filepath, on_split_missing = 'ignore');
    raw.load_data();
    raw.pick_types(meg=True, eeg=False, stim=True)

    ### EXCLUDING BAD CHANNELS ###
    # loading in the txt file with the channels that should be labeled as bad channels
    with open('../session_info.txt', 'r') as f:
        file = f.read()
        session_info = json.loads(file)

    # using dict[] notation to get the bad channels for the specific file. Not using dict.get() as this does not raise a key-error if the key does not exist
    list_bad_channels = session_info[filename]['bad_channels']

    # marking the channels as bad
    raw.info['bads'] = list_bad_channels

    ### CROPPING OF BEGGINNING AND ENDING OF MEG RECORDING ###
    tmin = session_info[filename]['tmin']
    tmax = session_info[filename]['tmax']
    cropped = raw.copy().crop(tmin = tmin, tmax = tmax)
    del raw


    ### BAND PASS FILTER ### 
    filt_raw = cropped.copy().filter(l_freq=1, h_freq=40)
    del cropped


    ### RESAMPLING ###
    resampled_raw = filt_raw.copy().resample(250)
    del filt_raw


    ### ICA ###
    ica = mne.preprocessing.ICA(n_components=None, random_state=97, method='fastica', max_iter=3000, verbose=None)
    ica.fit(resampled_raw)

    # saving the ICA solution
    ica.save(outpath, overwrite=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-in', '--infile', required=True, help='path to fif file')
    args = vars(ap.parse_args())
    main(args['infile'])
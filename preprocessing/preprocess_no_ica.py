"""
Script to preprocess the raw data and save the epochs (without using ICA to reject artefacts).
"""
import mne
import json
from pathlib import Path



if __name__ == '__main__':
    path = Path(__file__)

    # loading in the txt file with the channels that should be labeled as bad channels
    with open(path.parents[1] / 'info_files' / 'session_info.txt', 'r') as f:
        file = f.read()
        session_info = json.loads(file)

    sessions = session_info.keys()
    sessions = [session.split('.')[0] for session in sessions]

    for session in sessions:
        ### RAW DATA ###
        filepath_raw = f'/media/8.1/raw_data/franscescas_data/raw_data/{session}.fif'

        # Loading in the raw data
        raw = mne.io.read_raw_fif(filepath_raw, on_split_missing='warn');
        raw.load_data();
        raw.pick_types(meg=True, eeg=False, stim=True)

        ### EXCLUDING BAD CHANNELS ###
        # using dict[] notation to get the bad channels for the specific file. Not using dict.get() as this does not raise a key-error if the key does not exist
        bad_channels_file = session_info[session + '.fif']['bad_channels']

        # marking the channels as bad
        raw.info['bads'] = bad_channels_file

        # cropping beginning and end of recording
        tmin = session_info[session + '.fif']['tmin']
        tmax = session_info[session + '.fif']['tmax']

        cropped = raw.copy().crop(tmin = tmin, tmax = tmax)
        del raw

        ### HIGH PASS FILTERING ###
        filt_raw = cropped.copy().filter(l_freq=1, h_freq=40)
        del cropped

        filt_raw.interpolate_bads(origin=(0, 0, 0.04)) 

        # save no ICA epochs
        events = mne.find_events(filt_raw, shortest_event=1)
        reject = dict(grad=4000e-13, mag=4e-12)  

        # event ids
        with open('info_files/event_ids.txt', 'r') as f:
            file = f.read()
            event_ids = json.loads(file)

        #  creating the epochs
        epochs = mne.Epochs(filt_raw, events, event_ids, tmin=0, tmax=1, proj=True, baseline=None, preload=True, reject=reject, on_missing = 'warn')

        outpath = f'/media/8.1/final_data/laurap/epochs/{session}-no_ica-epo.fif'
        epochs = epochs.resample(250)
        epochs.save(outpath, overwrite = True)
"""
Generates files with the events for each session and saves them in the info_files folder.
"""

import os
import mne
import pickle as pkl


def generate_events_files(epoch_path):
    files = os.listdir(epoch_path)
    files = [file for file in files if not file.endswith('no_ica-epo.fif')]
    
    for file in files:
        epochs = mne.read_epochs(os.path.join(epoch_path, file))
        events = epochs.events
        events = events[:, 2]

        with open(os.path.join('events', f'{file.split("-epo.fif")[0]}_events.pkl'), 'wb') as f:
            pkl.dump(events, f)


if __name__ == '__main__':
    epoch_path = os.path.join(os.path.sep, 'media', '8.1', 'final_data', 'laurap', 'epochs')
    generate_events_files(epoch_path)

from pathlib import Path
from json import load
import numpy as np
import mne


# read session_info.txt
path = Path("info_files/session_info.txt")

with open(path, "r") as f:
    session_info = load(f)


sessions = [['visual_03', 'visual_04'], ['visual_05', 'visual_06', 'visual_07'], ['visual_08', 'visual_09', 'visual_10'], ['visual_11', 'visual_12', 'visual_13'],['visual_14', 'visual_15', 'visual_16', 'visual_17', 'visual_18', 'visual_19'],['visual_23', 'visual_24', 'visual_25', 'visual_26', 'visual_27', 'visual_28', 'visual_29'],['visual_30', 'visual_31', 'visual_32', 'visual_33', 'visual_34', 'visual_35', 'visual_36', 'visual_37', 'visual_38']]

# get all the tmins seperated by session

tmins = []
for session in sessions:
    tmins_session = []

    for sess in session:

        # read the raw file
        raw = mne.io.read_raw_fif(Path(f"/media/8.1/raw_data/franscescas_data/raw_data/{sess}.fif"), preload=True)

        # get the tmin
        tmin = session_info[f"{sess}.fif"]["tmin"]

        # crop the data
        #raw.crop(tmin=tmin, tmax=None)

        # get the time between the first sample and the first event in seconds
        events = mne.find_events(raw, shortest_event=1)
        tmin = (events[0][0] - raw.first_samp) / raw.info["sfreq"]

    
        tmins_session.append(tmin)
    
    tmins.append(tmins_session)


    
for i, tmin in enumerate(tmins):
    print(np.sum(tmin))



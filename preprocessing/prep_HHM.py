
import mne
import json
from pathlib import Path
from epochs_2_source_space import get_hpi_meg, rot3dfit, freesurfer_to_mri, transform_geometry, extract_labeled_timecourse, fsaverage_stcs
import scipy.io as sio
import numpy as np


if __name__ == "__main__":
    # loading in the txt file with the channels that should be labeled as bad channels
    with open('info_files/session_info.txt', 'r') as f:
        file = f.read()
        session_info = json.loads(file)

    outpath = Path('/media/8.1/final_data/laurap/rest_continuous_for_hmm')
    
    if outpath.exists() == False:
        outpath.mkdir(parents=True)

    data_dir = Path("/media/8.1")
    data_path = data_dir / "raw_data" / "franscescas_data" / "raw_data"
    src = mne.read_source_spaces(data_dir / "raw_data" / "franscescas_data" / "mri" / "sub1-oct6-src.fif")
    bem_sol = data_dir / "raw_data" / "franscescas_data" / "mri" / "subj1-bem_solution.fif"
    subject = 'subj1'
    subject_dir = data_dir / "raw_data" / "franscescas_data" / "mri"
    path_nii = data_dir / "scripts" / "laurap" / "franscescas_data" / "meg_headcast" / "mri" / "T1" / "sMQ03532-0009-00001-000192-01.nii"
    hpi_mri = sio.loadmat(str(data_dir / "scripts" / "laurap" / "franscescas_data" / "meg_headcast" / "hpi_mri.mat")).get('hpi_mri')

    for session in ['visual_03', 'visual_04', 'visual_05', 'visual_06', 'visual_07', 'visual_08', 'visual_09', 'visual_10', 'visual_11', 'visual_12', 'visual_13', 'visual_14', 'visual_15', 'visual_16', 'visual_17', 'visual_18', 'visual_19', 'visual_23', 'visual_24', 'visual_25', 'visual_26', 'visual_27', 'visual_28', 'visual_29', 'visual_30', 'visual_31', 'visual_32', 'visual_33', 'visual_34', 'visual_35', 'visual_36', 'visual_37', 'visual_38']:
        ### RAW DATA ###
        filepath_raw = data_path / f"{session}.fif"

        # Loading in the raw data
        raw = mne.io.read_raw_fif(filepath_raw, on_split_missing='warn');
        raw.load_data();
        raw.pick_types(meg=True, eeg=False, stim=True)

        ### EXCLUDING BAD CHANNELS ###
        # using dict[] notation to get the bad channels for the specific file. Not using dict.get() as this does not raise a key-error if the key does not exist
        # marking the channels as bad
        raw.info['bads'] = session_info[f"{session}.fif"]['bad_channels']

        ### HIGH PASS FILTERING ###
        filt_raw = raw.copy().filter(l_freq=1, h_freq=40)

        filt_raw.interpolate_bads(origin=(0, 0, 0.04)) 

        ### ICA ###
        ica_filename = f'/media/8.1/intermediate_data/laurap/ica/ica_solution/{session}-ica.fif'
        ica = mne.preprocessing.read_ica(ica_filename)
        ica.apply(filt_raw)


        noise_components = session_info[f"{session}.fif"]['noise_components']

        # removing the noise components
        filt_raw = ica.apply(filt_raw, exclude = noise_components)

        # crop the data from until the first event
        events = mne.find_events(filt_raw, shortest_event=1)

        # find the index for the first event with a trigger code between 1 and 118
        first_event = 0
        for i in range(len(events)):
            if events[i][2] in range(1, 119):
                first_event = i
                break

        tmax = (events[first_event][0] - filt_raw.first_samp) / filt_raw.info["sfreq"]

        filt_raw.crop(tmax=tmax)

        # downsample the data
        #filt_raw.resample(250)

        # transform the geometry of the sensors
        filt_raw = transform_geometry(filt_raw, hpi_mri, path_nii)

          
        fwd = mne.make_forward_solution(filt_raw.info, src = src, trans = None, bem = bem_sol)
        cov = mne.compute_raw_covariance(filt_raw, method='empirical') ## sample covariance is calculated
        inv = mne.minimum_norm.make_inverse_operator(filt_raw.info, fwd, cov, loose='auto')

        stc = mne.minimum_norm.apply_inverse_raw(filt_raw, inv, lambda2=1.0 / 3.0 ** 2, verbose=False, method="dSPM", pick_ori="normal")

        
        # extract label time courses
        label_time_course = extract_labeled_timecourse(stc, 'aparc', subject, subject_dir, inv['src'])
        np.save(outpath / f"{session}-parcelled", label_time_course)
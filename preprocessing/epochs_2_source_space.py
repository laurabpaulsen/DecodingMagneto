'''
To set up source space FreeSurfer was initially used for MRI reconstruction. See https://mne.tools/stable/auto_tutorials/forward/10_background_freesurfer.html#tut-freesurfer-reconstruction. 
MNE code used for setting up source space and creating BEM surfaces can be found in `source_space.py`. 

Usage, e.g., python epochs_2_source_space.py -s 'memory_01'


Dev notes:
- [ ] Add a function for extracting the labelled time courses
'''

import mne
import argparse
import scipy.io as sio
import numpy as np
import nibabel as nib
import os

def get_hpi_meg(epochs):
    hpi_coil_pos = np.array([dig['r'] for dig in epochs.info['hpi_results'][0]['dig_points']])
    
    # order of hpi coils is different in MEG and mri space, they are reordered here to match
    hpi_coil_pos[[0, 1, 2, 3]] = hpi_coil_pos[[2, 3, 1, 0]]
    
    return hpi_coil_pos

def rot3dfit(A, B):
    """
    Permforms a least-square fit for the linear form 
    Y = X*R + T

    where R is a 3 x 3 orthogonal rotation matrix, t is a 1 x 3
    translation vector, and A and B are sets of 3D points defined as
    3 x N matrices, where N is the number of points.

    Implementation of the rigid 3D transform algorithm from:
    Least-Squares Fitting of Two 3-D Point Sets,
    Arun, K. S. and Huang, T. S. and Blostein, S. D (1987)
    """
    assert A.shape == B.shape

    if A.shape[0] != 3 or B.shape[0] != 3:
        raise ValueError('A and B must be 3 x N matrices')

    # compute centroids (average points over each dimension (x, y, z))
    centroid_A = np.mean(A, axis=1) 
    centroid_B = np.mean(B, axis=1)
    
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # to find the optimal rotation both points are recentred
    # then both centroids are at the origin (subtract mean)
    Ac = A - centroid_A
    Bc = B - centroid_B

    # rotation matrix
    H = Ac @ Bc.T
    U, S, V = np.linalg.svd(H)
    R = V.T @ U.T

    # account for reflection
    if np.linalg.det(R) < 0:
        V[2,:] *= -1
        R = V.T @ U.T

    # translation vector
    t = -R @ centroid_A + centroid_B 

    # best fit 
    Yf = R @ A + t

    dY = B - Yf
    errors = []
    for point in range(dY.shape[0]):
        err = np.linalg.norm(dY[:, point])
        errors.append(err)

    print('Error: ', errors)
    return R, t, Yf

def freesurfer_to_mri(image_nii):
    '''
    The transformation to go from freesurfer space to mri space

    Parameters
    ----------
    image_nii : str
        path to the nifti file
    
    Returns
    -------
    translation : numpy.ndarray
        The translation matrix
    '''

    translation = np.eye(4)

    # load image
    image_nii = nib.load(image_nii)

    shape = np.array(image_nii.shape)
    center = shape / 2
    center_homogeneous = np.hstack((center, [1]))
    transform = image_nii.affine
    
    cras = (transform @ center_homogeneous)[:3]

    translation[:3, -1] = cras

    return np.linalg.inv(translation)

def transform_geometry(epochs, hpi_mri, image_nii):
    '''
    Changes the sensor positions and dev_head_t from device to mri

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object

    Returns
    -------
    epochs : mne.Epochs
        The epochs object with changed sensor positions and dev_head_t
    '''

    hpi_meg = get_hpi_meg(epochs)
    hpi_mri = hpi_mri/1000 # convert to meters

    
    # find rotation matrix and translation vector to move from MEG to MRI space
    R, T, yf = rot3dfit(hpi_meg.T, hpi_mri.T) # function needs 3 x N matrices

    meg_mri_t = np.zeros((4, 4))
    meg_mri_t[:3, :3] = R.T
    meg_mri_t[:3, 3] = T.T 
    meg_mri_t[3, 3] = 1


    # This transformation is used to go from MRI to freesurfer space
    trans = freesurfer_to_mri(image_nii = image_nii)
    trans[:3, -1] = trans[:3, -1]/1000
    epochs.info['dev_head_t']['trans'] = trans

    for i in range(len(epochs.info['chs'])):
        # change sensor positions
        location = epochs.info['chs'][i]['loc']
        loc = np.append(location[:3], 1)
        loc = loc @ meg_mri_t
        location[:3] = loc[:3]
        epochs.info['chs'][i]['loc'] = location
        # change sensor orientations
        rot_coils = np.array([location[3:6], location[6:9], location[9:12]])
        rot_coils = rot_coils @ R.T
        
        location[3:12] = rot_coils.flatten()

        if i ==305:
            break

    return epochs   

def extract_labeled_timecourse(stcs, parc, subject, subjects_dir, src): 
    """
    Extracts the timecourse of the labels in the stcs
    
    Parameters
    ----------
    stcs : list
        The list of stcs
    parc : str
        The parcellation to use
    subject : str
        The subject name
    subjects_dir : str
        The path to the subjects directory
    src : str
        The path to the source space file or source space object

    Returns
    -------
    label_time_course : list
        The list of label timecourses
    """
    
    labels_parc = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=subjects_dir)
    label_time_course = mne.extract_label_time_course(stcs, labels_parc, src, mode='pca_flip')

    return label_time_course

def fsaverage_stcs(stcs, subject_from, src_to, subjects_dir):
    """
    Morphs the stcs to fsaverage

    Parameters
    ----------
    stcs : list
        The list of stcs
    subject_from : str
        The subject name
    src_to : str
        The path to the source space file or source space object to morph to
    subjects_dir : str
        The path to the subjects directory
    
    Returns
    -------
    morphed_stcs : list
        The list of morphed stcs
    """
    
    morph = mne.compute_source_morph(stcs[0], subject_from = subject_from, subject_to='fsaverage', 
                                         src_to=src_to, subjects_dir=subjects_dir)
    
    morphed_stcs = [morph.apply(stc) for stc in stcs]

    return morphed_stcs
    

def main(session):
    src = mne.read_source_spaces(os.path.join(os.path.sep, 'media', '8.1', 'raw_data', 'franscescas_data', 'mri', 'sub1-oct6-src.fif'))
    bem_sol = os.path.join(os.path.sep, 'media', '8.1', 'raw_data', 'franscescas_data', 'mri', 'subj1-bem_solution.fif')
    subject = 'subj1'
    subject_dir = os.path.join(os.path.sep, 'media', '8.1', 'raw_data', 'franscescas_data', 'mri')
    epoch_path = os.path.join(os.path.sep, 'media', '8.1', 'final_data', 'laurap', 'epochs', f'{session}-epo.fif')
    path_nii = os.path.join(os.path.sep, 'media', '8.1', 'scripts', 'laurap', 'franscescas_data', 'meg_headcast', 'mri', 'T1', 'sMQ03532-0009-00001-000192-01.nii')
    hpi_mri = sio.loadmat(os.path.join(os.path.sep, 'media', '8.1', 'scripts', 'laurap', 'franscescas_data', 'meg_headcast', 'hpi_mri.mat')).get('hpi_mri')

    epochs = mne.read_epochs(epoch_path)
    epochs = transform_geometry(epochs, hpi_mri, path_nii)
    
    fwd = mne.make_forward_solution(epochs.info, src = src, trans = None, bem = bem_sol)
    cov = mne.compute_covariance(epochs, method='empirical') ## sample covariance is calculated
    inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, loose='auto')
    
    # applying the inverse solution to the epochs
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2=1.0 / 3.0 ** 2, verbose=False, method="dSPM", pick_ori="normal")
    stcs_array = np.array([stc for stc in stcs])
    np.save(f'/media/8.1/final_data/laurap/source_space/sources/{session}_source', stcs_array)

    # extract label time courses for three of the cortical parcellations
    for parc in ['aparc.a2009s', 'aparc', 'aparc.DKTatlas']:
        label_time_course = extract_labeled_timecourse(stcs, parc, subject, subject_dir, inv['src'])
        np.save(f'/media/8.1/final_data/laurap/source_space/parcelled/{parc}/{session}_parcelled', label_time_course)

    
    # extract hcpmmp1 time courses
    fname_fsaverage_src = (os.path.join(subject_dir, 'fsaverage','bem', 'fsaverage-ico-5-src.fif'))
    src_to = mne.read_source_spaces(fname_fsaverage_src)

    morphed_stcs = fsaverage_stcs(stcs, subject, src_to, subject_dir)

    src_fs = mne.read_source_spaces('/media/8.1/raw_data/franscescas_data/mri/fsaverage/bem/fsaverage-ico-5-src.fif')
    label_time_course = extract_labeled_timecourse(morphed_stcs, 'HCPMMP1', 'fsaverage', subject_dir, src_fs)
    
    np.save(f'/media/8.1/final_data/laurap/source_space/parcelled/HCMMP1/{session}_parcelled', label_time_course)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--session', required=True, help='session, e.g., visual_03')
    args = ap.parse_args()

    main(args.session)


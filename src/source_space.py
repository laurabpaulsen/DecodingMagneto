'''
This file holds the code for setting up the source space as well as the boundary-element model. 

To set up source space FreeSurfer was initially used for MRI reconstruction. 
See https://mne.tools/stable/auto_tutorials/forward/10_background_freesurfer.html#tut-freesurfer-reconstruction
'''
import sys
sys.path.append('/home/laurap/.local/bin')
import mne

src = mne.setup_source_space('subj1', spacing='oct6', subjects_dir='/media/8.1/francescas_data/mri')
mne.write_source_spaces('/media/8.1/francescas_data/mri/sub1-oct6-src.fif', src) 

# setting up boundary-element model (BEM)
## remember to create BEM surfaces before mne watershed_bem -s subj1 -d  /media/8.1/francescas_data/mri
model = mne.make_bem_model('subj1', subjects_dir='/media/8.1/francescas_data/mri')  
mne.write_bem_surfaces('/media/8.1/francescas_data/mri/subj1-bem.fif', model)
bem_sol = mne.make_bem_solution(model, verbose=None)
mne.write_bem_solution('/media/8.1/francescas_data/mri/subj1-bem_solution.fif', bem_sol, overwrite=False, verbose=None)
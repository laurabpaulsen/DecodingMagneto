# DecodingMagneto

## Data
The data consists of MEG data from a single subject. The participant was presented with visual stimuli (118 different images). 

The participant was subject to two different conditions:
- **Condition 1**: The participant was presented with the images and only had to attend to the images.
- **Condition 2**: The participant was presented with the images and had to perform a memory task.

Regardless of the condition, no behavioral data was collected.

## Project Organization
```
├── decoding
│   ├── cross_decoding                  <- Directory containing the cross decoding analysis
│   │    ├── accuracies                 <- Directory for saving the accuracies
│   │    │   ├── cross_decoding_10_LDA_aparc.npy       
│   │    │   └── ...
│   │    ├── plots                      <- Directory for saving the plots
│   │    └── cross_decoding.py          <- Runs cross decoding analysis 
├── info_files                          <- Files containing information about the data
│   ├── events                          <- Directory containing the events for each session 
│   │    ├── memory_01_events.pkl       <- Pickle file containing the events for the memory task in session 01
│   │    └── ...
│   ├── hpi_mri.mat                     <- Mat file containing the MRI positions of the HPI
│   ├── event_ids.txt                   <- Mapping of the stimuli to the triggers
│   ├── event_session_info.py           <- Creates event_ids.txt and session_info.py
│   ├── generate_files_events.py        <- script for generating the events files
│   └── session_info.txt                <- Bad channels, ICA noise components, etc for each session
├── src                                 <- Scripts for preprocessing of the data
│   ├── check_ica.ipynb                 <- Plotting of the ICA components
│   ├── check_raw.ipynb                 <- Plotting of the raw data for identifying bad channels, tmin and tmax
│   ├── epochs_2_source_space.py        <- Script for source reconstruction
│   ├── preprocces_no_ica.py            <- Script for preprocessing without ICA
│   ├── run_ica.ipynb                   <- Running ICA on the data
│   └── source_space.py                 <- Setting up source space and BEM
├── helper_functions.py                 <- Helper functions for the scripts
├── plot_functions.py                   <- Plotting functions for the scripts
└── README.md                           <- The top-level README for this project.  
```
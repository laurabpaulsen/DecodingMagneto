
## Pipeline
The files found in this folder were used to preprocess the data. The following steps were taken:

| File | Purpose | Notes |
|-----------|:------------|:--------|
```check_raw.ipynb``` | Identify bad channels, tmin and tmax | Bad channels, tmin and tmax added to ```event_session_info.py```
 ```run_ica.py``` |Run ICA |
```check_ica.ipynb``` |Identifying noise components and epoching |  Noise components to ```event_session_info.py```
```epochs_2_source_space.py``` | Source reconstruction | 

## Files in the preprocessing directory
```
├── check_ica.ipynb                 <- Plotting of the ICA components
├── check_raw.ipynb                 <- Plotting of the raw data for identifying bad channels, tmin and tmax
├── epochs_2_source_space.py        <- Script for source reconstruction
├── preprocces_no_ica.py            <- Script for preprocessing without ICA
├── run_ica.ipynb                   <- Running ICA on the data
└── source_space.py                 <- Setting up source space and BEM
```


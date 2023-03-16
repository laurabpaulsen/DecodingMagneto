# DecodingMagneto

## Data
The data consists of MEG data from a single subject. The participant was presented with visual stimuli (118 different images). 

The participant was subject to two different conditions:
- **Condition 1**: The participant was presented with the images and only had to attend to the images.
- **Condition 2**: The participant was presented with the images and had to perform a memory task.

Regardless of the condition, no behavioral data was collected.

## Usage
1. Clone the repository
2. ??? download the data ???
3. Create environment and install dependencies
```
setup.sh
```


## Project Organization
```
├── decoding
│   ├── cross_decoding                  <- Decoding across sessions
│   │    ├── accuracies                
│   │    │   ├── cross_decoding_10_LDA_aparc.npy       
│   │    │   └── ...
│   │    ├── plots                      
│   │    └── cross_decoding.py          
│   ├── condition                       <- Decoding between conditions (memory vs. no memory)
│   └── ica                             <- Decoding each session with and without ICA components removed
├── info_files                           
├── preprocessing        
├── utils                               <- Local modules
│   ├── __init__.py
│   ├── data                            <- Functions for loading and preprocessing the data
│   └── analysis                        <- Functions for decoding, plotting, etc
└── README.md                           <- The top-level README for this project.  
```
# Decoding
This folder holds the subfolders for the different decoding analyses. The following analyses were performed:
| Directory | Purpose 
|:-----------|:------------
| condition | Decoding the task condition (memory vs. visual) 
| cross_decoding | Decoding animacy training and testing on all pairs of sessions
| cross_bins | mimicking the cross decoding analysis but instead of training on all pairs of sessions, training on all pairs of bins (each session is split into 11 bins)
| ica | Testing the effect of ICA on decoding

## Files in the decoding directory
```
├── condition
│   ├── accuracies
│   ├── condition_decoding.py
│   ├── plot_results.py
│   └── ...
├── cross_decoding
│   ├── accuracies
│   ├── cross_decoding.py
│   ├── plot_results.py
│   └── ...
├── ica
│   ├── accuracies
│   ├── ica_decoding.py
│   ├── plot_results.py
│   └── ...
└── README.md
```
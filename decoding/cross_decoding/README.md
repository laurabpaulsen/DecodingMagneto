# cross_decoding

You can modify the parameters for the cross-decoding analysis by parsing arguments when running the script from the command line. Run the following command to see the available arguments:
```
python cross_decoding.py -h
```

For example if you want to run the cross decoding analysis with the `aparc` parcellation, the `LDA` classifier and 10 cores, you can run the following command:
```
python cross_decoding.py --parcellation aparc --classifier LDA --n_jobs 10
```

The results will be saved in the folder `accuracies` as a `.npy` file.
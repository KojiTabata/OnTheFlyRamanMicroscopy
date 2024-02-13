# on-the-fly-Raman

This repository contains python code for validating algorithms to accelerate diagnosis by Raman measurements. 

The sciprt written in script/run_experiment.py takes as input a pickle file that stores a precomputed malignancy index distribution and conduct simulation experiments to compare several multi-armed bandit algorithms.
pickle file 
The pickle file names are positive data sample data/anom_index/FTC133_{01,02,03,04,05}.pickle and negative data file data/anom_index/Nthyori31_{01,02,03,04}.pickle.
The information stored in the pickle file is a 3-dimensional numpy array.
The probability that the Raman spectrum at position (x, y) in the space of the sample to be measured is the background, normal, and abnormal are the [x, y, 0], [x, y, 1], corresponds to the component values of [x, y, 2].

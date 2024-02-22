# Dataset and Code for 'On-the-Fly Raman Microscopy Guaranteeing the Accuracy of Discrimination'

This repository contains python code for validating algorithms to accelerate diagnosis by Raman measurements. 

The script written in script/run_experiment.py takes as input a pickle file that stores a precomputed distribution of anomaly index and conduct simulation experiments to compare several multi-armed bandit algorithms.
The pickle file names are data/anom_index/FTC133_{01,02,03,04,05}.pickle for positive data sample and data/anom_index/Nthyori31_{01,02,03,04}.pickle for negative data file. 
The information stored in the pickle files is a 3-dimensional numpy array.
The probabilities that the Raman spectrum at position (x, y) in the space of the sample to be measured is estimated as the background, normal, and abnormal are corresponds to the elements of array at [x, y, 0], [x, y, 1], [x, y, 2], respectively. 
Threreofre, anomaly index is  [x, y, 2]. 


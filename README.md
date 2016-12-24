# sista-rnn
Code for the paper
S. Wisdom, T. Powers, J. Pitton, and L. Atlas, “Interpretable Recurrent Neural Networks Using Sequential Sparse Recovery,” arXiv preprint arXiv:1611.07252, 2016.

Based on code by 
Salman Asif, available from https://github.com/sasif/L1-homotopy
Martin Arjovsky, Amar Shah, and Yoshua Bengio, avaiable from github.com/amarshah/complex_RNN

First, download the Caltech-256 dataset available from http://www.vision.caltech.edu/Image_Datasets/Caltech256/

Code for replicating unsupervised results coming soon.

To replicate the supervised results, follow these steps:
1) Execute the 'run_supervised.sh' script. Make sure you change 'path_dataset' in the YAML configuration files to the path of the Caltech-256 dataset on your machine. Note that MSEs reported during training are not the same as those reported in the paper, since the images are normalized during training.
2) Score the results using matlab/score_caltech256.m and matlab/print_scores_table_caltech256.m

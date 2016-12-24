# sista-rnn
Code for the paper
S. Wisdom, T. Powers, J. Pitton, and L. Atlas, “Interpretable Recurrent Neural Networks Using Sequential Sparse Recovery,” arXiv preprint arXiv:1611.07252, 2016.

Based on code by 
Salman Asif, available from https://github.com/sasif/L1-homotopy
Martin Arjovsky, Amar Shah, and Yoshua Bengio, avaiable from github.com/amarshah/complex_RNN

First, download the Caltech-256 dataset available from http://www.vision.caltech.edu/Image_Datasets/Caltech256/

To replicate the paper's results, follow these steps:
1) Execute the 'run_supervised.sh' script. This will load and preprocess the Caltch-256 dataset for all other functions, as well as training the supervised models. Make sure you change 'path_dataset' in the YAML configuration files in the 'python' directory to the path of the Caltech-256 dataset on your machine. Note that MSEs reported during training are not the same as those reported in the paper, since the images are normalized during training. The outputs of the networks on the test set will be stored in the 'caltech256' folder.
2) Compile MEX files for unsupervised baselines by running 'compile.m' from 'matlab/L1-homotopy'.
3) From the 'matlab' directory, run the unsupervised baselines using 'matlab/run_baselines_caltech256_v2.m'. This will also generate reference images in the 'caltech256' directory.
4) From the 'matlab' directory, score the results using 'matlab/score_caltech256.m' and print a table of the results using 'matlab/print_scores_table_caltech256.m'


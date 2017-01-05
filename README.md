# sista-rnn
Code for the papers

S. Wisdom, T. Powers, J. Pitton, and L. Atlas, “Building Recurrent Networks by Unfolding Iterative Thresholding for Sequential Sparse Recovery,” ICASSP 2017, New Orleans, LA, USA, March 2017

S. Wisdom, T. Powers, J. Pitton, and L. Atlas, “Interpretable Recurrent Neural Networks Using Sequential Sparse Recovery,” arXiv preprint arXiv:1611.07252, 2016. Presented at NIPS 2016 Workshop on Interpretable Machine Learning in Complex Systems, Barcelona, Spain, December 2016



## Includes code by:

Stephen J. Wright, Robert D. Nowak, and Mario Figueiredo, available from https://www.lx.it.pt/~mtf/SpaRSA/SpaRSA_2.0.zip

Salman Asif, available from https://github.com/sasif/L1-homotopy

Martin Arjovsky, Amar Shah, and Yoshua Bengio, avaiable from https://github.com/amarshah/complex_RNN



## To replicate the paper's results, follow these steps:

1) Download the Caltech-256 dataset available from http://www.vision.caltech.edu/Image_Datasets/Caltech256/

2) Execute the 'run_supervised.sh' script. This will load and preprocess the Caltch-256 dataset for all other functions, as well as training the supervised models. Make sure you change 'path_dataset' in the YAML configuration files in the 'python' directory to the path of the Caltech-256 dataset on your machine. Note that MSEs reported during training are not the same as those reported in the paper, since the images are normalized during training. The outputs of the networks on the test set will be stored in the 'caltech256' folder.

3) Compile MEX files for unsupervised baselines by running 'compile.m' from 'matlab/L1-homotopy'.

4) From the 'matlab' directory, run the unsupervised baselines using 'matlab/run_baselines_caltech256_v2.m'. This will also generate reference images in the 'caltech256' directory.

5) From the 'matlab' directory, score the results using 'matlab/score_caltech256.m' and print a table of the results using 'matlab/print_scores_table_caltech256.m'


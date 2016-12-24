#!/bin/bash

#LSTM
THEANO_FLAGS="device=gpu0" python2.7 python/train_rnn.py '' -c python/config_caltech256_3layers_LSTM_ds1_N128_100epochs.yaml

#generic RNN with soft-thresholding nonlienarity
THEANO_FLAGS="device=gpu0" python2.7 python/train_rnn.py '' -c python/config_caltech256_3layers_v3_lam0-5_ds1_N128_latticeNoInputConn_randomInit_100epochs.yaml

#SISTA-RNN
THEANO_FLAGS="device=gpu0" python2.7 python/train_rnn.py '' -c python/config_caltech256_3layers_v3_lam0-5_ds1_N128_100epochs_trainsista.yaml


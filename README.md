# Multi-view-neural-acoustic-words-embeddings
# Prerequisites
Python --- 2.7.10, Tensorflow --- 0.9.0rc0, numpy, scipy
# How to use
To help organizing the code, here is my working directory

    /Multi-view_embedding
        /Data
        /Outputs
        /Saved_Models
        /Scripts

To train a model with 2-layer bidirectional LSTMs on both views:

    python 2biLSTM.py <parameters> (e.g. python 2biLSTM.py -lr 0.0001 -obj 02 -m 0.4)
    available parameters include:
      -lr: learning rate, 0.001 by default (for LSTMs with more than 1 layer, 0.0001 is recommended)
      -obj: objectives, 0(obj0) by default, 02 = averaged sum of obj0 and obj2
      -hs: hidden size, 512 by default
      -m: margin, 0.5 by default
      -kp: keep probability of dropout, 0.6 by default
      -bs: batch size, 20 by default
      -epo: epochs, 20 by default
      -inits: initialization scale of LSTM cells, 0.05 by default

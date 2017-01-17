# Multi-view-neural-acoustic-words-embeddings
Prerequisites: Python --- 2.7.10, Tensorflow --- 0.9.0rc0, numpy, scipy

To train a model with 2-layer bidirectional LSTMs on both views:

    2biLSTM.py <parameters>
    available parameters include:
      -lr: learning rate, 0.001 by default
      -obj: objectives, 0(obj0) by default, 02 = averaged sum of obj0 and obj2
      -hs: hidden size, 512 by default
      -m: margin, 0.5 by default
      -kp: keep probability of dropout, 0.6 by default
      -bs: batch size, 20 by default
      -epo: epochs, 20 by default
      -inits: initialization scale of LSTM cells, 0.05 by default

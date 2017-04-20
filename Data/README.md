# Data preparation
3 files:
trainset.list shows the data for train set, indicating which word, which conversation in SwitchBoard and which timeframes(in such order).

devset.list for development set.

testset.list for test set.

Please feel free to implement your own way if you think this is not convenient enough.

In the case you want to identically rerun my code, here is my Data directory:

    /Data
        train.npz
        dev.npz
        test.npz
        input1.npy
        input2.npy

Each .npz file corresponds to a dataset. The file is actually a dictionary, the key is the word, the value is acoustic sequence feature data with a shape of len * 39(MFCCs).

input1.npy is an array of values of train.npz

input2.npy is an array of character embeddings, in which each element is a len * 26(num of english letters) matrix corresponding to a word.

Note that input1.npy and input2.npy must be of same order, i.e. input1[0] is the acoustic feature of word input2[0].
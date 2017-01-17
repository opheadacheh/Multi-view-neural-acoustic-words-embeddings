import numpy as np
import random, time, operator
import samediff
import tensorflow as tf
import argparse
import os
biLSTM2 = __import__('2biLSTM')


def Padding(data):
    lengths = []
    for matrix in data:
        lengths.append(matrix.shape[0])
    max_len = np.max(lengths)
    Pdata = []
    for matrix in data:
        Pdata.append(np.pad(matrix, ((0, max_len - matrix.shape[0]), (0, 0)), mode='constant', constant_values=0))
    return np.asarray(Pdata), lengths


def word2onehot(word):
    word = word.replace('{', '').replace('}', '').replace('-', '').replace('\'', '')
    mat = np.zeros((len(word), 26), dtype=int)
    for i in range(len(word)):
        idx = ord(word[i]) - 97
        mat[i, idx] = 1
    return mat


def GetUnmatchedAcous(data, config):
    input_x2 = []
    labels = data.files
    for label in labels:
        name = label[: label.find('_')]
        otherlabel = random.choice(labels)
        othername = otherlabel[: otherlabel.find('_')]
        while othername == name:
            otherlabel = random.choice(labels)
            othername = otherlabel[: otherlabel.find('_')]
        input_x2.append(data[otherlabel])
    input_x2_set = []
    input_x2_lengths_set = []
    now = 0
    while now < len(labels):
        d, lengths = Padding(input_x2[now: now + config.batch_size])
        input_x2_set.append(d)
        input_x2_lengths_set.append(lengths)
        now += config.batch_size
    return input_x2_set, input_x2_lengths_set


def GetUnmatchedLabel(labels, names, config, return_othernames=False):
    input_c2 = []
    othernames = []
    for label in labels:
        name = label[: label.find('_')]
        othername = name
        while othername == name:
            othername = random.choice(names)
        othernames.append(othername)
        input_c2.append(word2onehot(othername))
    input_c2_set = []
    input_c2_lengths_set = []
    now = 0
    while now < len(labels):
        data, lengths = Padding(input_c2[now: now + config.batch_size])
        input_c2_set.append(data)
        input_c2_lengths_set.append(lengths)
        now += config.batch_size
    if not return_othernames:
        return input_c2_set, input_c2_lengths_set
    else:
        return input_c2_set, input_c2_lengths_set, othernames


def GetUnmatchedLabel_onehot(labels, names, config):
    input3 = np.zeros((len(labels), len(names)), dtype=int)
    for i, label in enumerate(labels):
        name = label[: label.find('_')]
        othername = name
        while othername == name:
            othername = random.choice(names)
        input3[i, names.index(othername)] = 1
    input3_set = []
    now = 0
    while now < len(labels):
        data = input3[now: now + config.batch_size]
        input3_set.append(data)
        now += config.batch_size
    return input3_set


def GetNames():
    names = []
    train = np.load('../Data/train.npz')
    for i in train:
        name = i[: i.find('_')]
        if name not in names:
            names.append(name)
    return names


def GetMoreNames():
    names = []
    train = np.load('../Data/train.npz')
    for i in train:
        name = i[: i.find('_')]
        if name not in names:
            names.append(name)

    test = np.load('../Data/test.npz')
    for i in test:
        name = i[: i.find('_')]
        if name not in names:
            names.append(name)

    dev = np.load('../Data/dev.npz')
    for i in dev:
        name = i[: i.find('_')]
        if name not in names:
            names.append(name)
    return names


def GetOneHot():
    names = GetNames()
    size = len(names)
    train = np.load('../Data/train.npz')
    mat = np.zeros((len(train.files), size), dtype=int)
    for i, label in enumerate(train):
        label = label[: label.find('_')]
        mat[i, names.index(label)] = 1
    return mat


def GetTestData(dict):
    labels = []
    data = []
    for i in dict.files:
        labels.append(i[: i.find('_')])
        data.append(dict[i])
    data, lengths = Padding(data)
    matches = samediff.generate_matches_array(labels)
    return data, lengths, matches


def GetTextData(dict):
    data = []
    for i in dict.files:
        data.append(word2onehot(i[: i.find('_')]))
    data, lengths = Padding(data)
    return data, lengths


def GetInputData(data_path, config):
    x1 = np.load(data_path + 'input1.npy')
    c1 = np.load(data_path + 'input2.npy')
    input1_set = []
    input2_set = []
    input1_lengths_set = []
    input2_lengths_set = []
    now = 0
    while now < len(x1):
        data, lengths = Padding(x1[now: now + config.batch_size])
        input1_set.append(data)
        input1_lengths_set.append(lengths)
        data, lengths = Padding(c1[now: now + config.batch_size])
        input2_set.append(data)
        input2_lengths_set.append(lengths)
        now += config.batch_size
    return input1_set, input2_set, input1_lengths_set, input2_lengths_set


def GetInputData_onehot(data_path, config):
    x1 = np.load(data_path + 'input1.npy')
    c1 = GetOneHot()
    input1_set = []
    input2_set = []
    input1_lengths_set = []
    now = 0
    while now < len(x1):
        data, lengths = Padding(x1[now: now + config.batch_size])
        input1_set.append(data)
        input1_lengths_set.append(lengths)
        data = c1[now: now + config.batch_size]
        input2_set.append(data)
        now += config.batch_size
    return input1_set, input2_set,  input1_lengths_set


def GetDisplayData(data_path):
    data = np.load(data_path)
    labels = []
    output_data = []
    for i in data:
        labels.append(i[: i.find('_')])
        output_data.append(data[i])
    return labels, output_data


def EditDistance(a, b):
    assert type(a) == type('str')
    assert type(b) == type('str')
    c = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    for i in range(1, len(a) + 1):
        c[i][0] = i
    for i in range(1, len(b) + 1):
        c[0][i] = i
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                c[i][j] = c[i-1][j-1]
            else:
                c[i][j] = 1000
            c[i][j] = min(c[i-1][j] + 1, c[i][j-1] + 1, c[i-1][j-1] + 1, c[i][j])
    return c[len(a), len(b)]


def GetTextEmbedding(words, conf, model_type, model_path):
    data = []
    for word in words:
        data.append(word2onehot(word))
    pdata, lengths = Padding(data)
    conf.keep_prob = 1.0
    with tf.variable_scope('model'):
        m = model_type(True, conf)
    saver = tf.train.Saver(tf.all_variables())
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        vdata = sess.run(m.word_state, {m.input_c1: pdata,
                                        m.input_c1_lengths: lengths})
    return vdata


def GetAcousticEmbedding(data, conf, model_type, model_path):
    vdata_set = []
    now = 0
    conf.keep_prob = 1.0
    with tf.variable_scope('model'):
        m = model_type(True, conf)
    saver = tf.train.Saver(tf.all_variables())
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        while now < len(data):
            pdata, lengths = Padding(data[now: now + conf.eval_batch_size])
            vdata = sess.run(m.final_state, {m.input_x1: pdata,
                                             m.input_x1_lengths: lengths})
            vdata_set.append(vdata)
            now += conf.eval_batch_size
        vdata_set = np.vstack(vdata_set)
    return vdata_set


def GetAcousticEmbeddingAsDict(data_path, model_path, model_type, model_conf, save_path=None):
    data = np.load(data_path)
    acous_data = []
    for label in data:
        acous_data.append(data[label])
    with tf.variable_scope('model'):
        m = model_type.SimpleLSTM(False, model_conf)
    embed_data = GetAcousticEmbedding(acous_data, model_conf, m, model_path)
    dict = {}
    for i, label in enumerate(data.files):
        dict[label] = embed_data[i]
    if save_path != None:
        np.savez(save_path, **dict)


def GetLosses(m, sess, input_x1_set, input_x2_set, input_c1_set, input_c2_set, input_x1_lengths_set, input_x2_lengths_set, input_c1_lengths_set, input_c2_lengths_set, labels1, labels2):
    losses = []
    dict = {}
    num_mini_batches = len(input_x1_set)
    num = 0
    for i in range(num_mini_batches):
        i = 389
        loss = sess.run(m.loss, {m.input_x1: input_x1_set[i],
                                 m.input_c1: input_c1_set[i],
                                 m.input_c2: input_c2_set[i],
                                 m.input_x1_lengths: input_x1_lengths_set[i],
                                 m.input_c1_lengths: input_c1_lengths_set[i],
                                 m.input_c2_lengths: input_c2_lengths_set[i]})
        losses.append(loss)
        if loss > 0.0:
            num += 1
            dict[str(i) + '-' + labels1[i] + '-' + labels2[i]] = loss
    dict = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
    fout = open('../Data4Display/Loss_Edis_stats.txt', 'w+')
    print dict
    for key, value in dict:
        fout.write(key + ' ' + str(value) + '\n')
    fout.write('total number of losses greater than 0: ' + str(num))
    return losses


def LossVsEditDis(model_path, model_type, model_conf, save_path):
    model_conf.batch_size = 1
    train = np.load('../Data/train.npz')
    labels = []
    for i in train.files:
        label = i[:i.find('_')]
        labels.append(label)
    print labels[389]
    for i, label in enumerate(train):
        if i == 389:
            print label
    names = GetNames()
    input_x1_set, input_c1_set, input_x1_lengths_set, input_c1_lengths_set = GetInputData('../Data/', model_conf)
    input_c2_set, input_c2_lengths_set, otherlabels = GetUnmatchedLabel(train.files, names, model_conf, True)
    with tf.variable_scope('model'):
        m = model_type.SimpleLSTM(True, model_conf)
    saver = tf.train.Saver(tf.all_variables())
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        losses = GetLosses(m, sess, input_x1_set, None, input_c1_set, input_c2_set, input_x1_lengths_set, None, input_c1_lengths_set, input_c2_lengths_set, labels, otherlabels)
    edit_distances = []
    for i in range(len(otherlabels)):
        edit_distances.append(EditDistance(labels[i], otherlabels[i]))
    dict = {}
    dict['loss'] = losses
    dict['edit_distance'] = edit_distances
    np.savez(save_path, **dict)
    return


def OccCount(labels):
    dict = {}
    for label in labels:
        if label in dict.keys():
            dict[label] += 1
        else:
            dict[label] = 1
    return dict


def Pack(list, conf):
    now = 0
    set = []
    while now < len(list):
        set.append(list[now: now + conf.batch_size])
        now += conf.batch_size
    return set


def GetPhoneticsDict():
    f = open('en_lexic.v03')
    content = f.read().split('\n')
    content = content[14: -1]
    dict = {}
    for i in range(len(content)):
        l = content[i].split('\t')
        dict[l[0]] = l[1].replace('+', '').replace('.', '').replace('\'', '')
    return dict


def AllEditDistances(full_labels, dataname):
    labels = []
    for label in full_labels:
        labels.append(label[: label.find('_')])
    print labels
    edit_distances = []
    for i in range(len(labels)):
        print i
        for j in range(i+1, len(labels)):
            edit_distances.append(EditDistance(labels[i], labels[j]))
    np.save('../Data4Display/' + dataname + '.npy', edit_distances)


def BestAP(logfile):
    contents = open(logfile).read().split('\n')
    best_AP = 0
    idx_now = -4
    idx = -4
    for content in contents:
        if content.startswith('Dev'):
            idx_now += 5
            t = float(content[content.find(': ') + 2: ])
            if t > best_AP:
                best_AP = t
                idx = idx_now
    return best_AP, idx


def ModelClean(indices, path, output_name):
    for file in os.listdir(path):
        if file.startswith(output_name):
            version = file[file.find('-') + 1:]
            if version.endswith('meta'):
                version = version[:-5]
            version = int(version)
            if version not in indices:
                os.remove(path + '/' + file)


def ArgsHandle():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', default='0.001', help='learning rate')
    parser.add_argument('-obj', default='0', help='objective')
    parser.add_argument('-hs', default='512', help='LSTM hidden size')
    parser.add_argument('-m', default='0.5', help='fixed margin')
    parser.add_argument('-kp', default='0.6', help='keep probability')
    parser.add_argument('-bs', default='20', help='batch size')
    parser.add_argument('-epo', default='20', help='epochs')
    parser.add_argument('-inits', default='0.05', help='LSTM cell initialization scale')
    parser.add_argument('-lastepoch', default='-1', help='train from which epoch, -1 means scratch')
    parser.add_argument('-th', default='11', help='threshold for sensitive margin')
    return parser.parse_args()


def ModelName(args):
    return args.obj + '_' + args.hs + '_' + args.m + '_' + args.lr + '_' + args.kp + '_' + args.bs + '_' + args.epo + '_' + args.inits + '-' + args.lastepoch


def OutputName(args):
    return args.obj + '_' + args.hs + '_' + args.m + '_' + args.lr + '_' + args.kp + '_' + args.bs + '_' + args.epo + '_' + args.inits


def main():
    args = ArgsHandle()
    GetTextEmbedding(['employee', 'employees'], biLSTM2.Config(args.obj, args.hs, args.m, args.lr, args.kp, args.bs, args.epo, args.inits), biLSTM2.SimpleLSTM, '../Saved_Models/2biLSTM_models/4_512_0.4_0.0001_0.6-999')
    pass

if __name__ == '__main__':
    main()
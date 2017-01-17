LSTM = __import__('LSTM')
biLSTM = __import__('biLSTM')
biLSTM2 = __import__('2biLSTM')
biLSTM2M= __import__('2biLSTM+margin')
import sys, time, samediff
import numpy as np
import data_processing as dp
import tensorflow as tf
from scipy.spatial.distance import pdist, cosine
import random


def acous_text_eval(m, sess, data, lengths, text_data, text_lengths, matches, config):
    embeddings = []
    now = 0
    while now < len(data):
        embedding = sess.run(m.final_state, {m.input_x1: data[now: now + config.eval_batch_size],
                                             m.input_x1_lengths: lengths[now: now + config.eval_batch_size]})
        embeddings.append(embedding)
        now += config.eval_batch_size
    X = np.vstack(embeddings)
    text_embeddings = []
    now = 0
    while now < len(data):
        text_embedding = sess.run(m.word_state, {m.input_c1: text_data[now: now + config.eval_batch_size],
                                                 m.input_c1_lengths: text_lengths[now: now + config.eval_batch_size]})
        text_embeddings.append(text_embedding)
        now += config.eval_batch_size
    Y = np.vstack(text_embeddings)
    distances = []
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            distances.append(cosine(X[i], Y[j]))
    distances = np.asarray(distances)
    ap, prb = samediff.average_precision(distances[matches == True], distances[matches == False])
    print "Average precision:", ap
    print "Precision-recall breakeven:", prb
    return ap


def cosine_distance(m, sess, data, lengths, text_data, text_lengths, matches, config, full_labels):
    labels = []
    for label in full_labels:
        labels.append(label[: label.find('_')])

    dict = dp.GetPhoneticsDict()

    phonetics = []
    for i, label in enumerate(labels):
        label = label.replace('{', '').replace('}', '')
        if label not in dict.keys():
            phonetics.append('')
        else:
            phonetics.append(dict[label])

    embeddings = []
    now = 0
    while now < len(data):
        embedding = sess.run(m.final_state, {m.input_x1: data[now: now + config.eval_batch_size],
                                             m.input_x1_lengths: lengths[now: now + config.eval_batch_size]})
        embeddings.append(embedding)
        now += config.eval_batch_size
    X = np.vstack(embeddings)
    text_embeddings = []
    now = 0
    while now < len(data):
        text_embedding = sess.run(m.word_state, {m.input_c1: text_data[now: now + config.eval_batch_size],
                                                 m.input_c1_lengths: text_lengths[now: now + config.eval_batch_size]})
        text_embeddings.append(text_embedding)
        now += config.eval_batch_size
    Y = np.vstack(text_embeddings)
    acous_distances = []
    text_distances = []
    edit_distances = []
    phone_distances = []
    for i in range(len(data)):
        if phonetics[i] == '':
            continue
        idx = [i]
        while len(idx) < 11:
            j = random.choice(range(len(data)))
            if j in idx or phonetics[j] == '':
                continue
            edit_distances.append(dp.EditDistance(labels[i], labels[j]))
            phone_distances.append(dp.EditDistance(phonetics[i], phonetics[j]))
            acous_distances.append(cosine(X[i], X[j]))
            text_distances.append(cosine(Y[i], Y[j]))
            idx.append(j)
    dict = {}
    dict['acous_distances'] = acous_distances
    dict['text_distances'] = text_distances
    dict['edit_distances'] = edit_distances
    dict['phone_distances'] = phone_distances
    np.savez('../Data4Display/m.5_th9_acous_text_phone_edit_dis.npz', **dict)


def main():
    model_name = '4_512_0.4_0.0001_0.6-999'
    data_name = 'test.npz'
    # params = model_name.split('_')
    # params[4] = params[4][: params[4].find('-')]

    start = time.time()
    data = np.load('../Data/' + data_name)
    # text_data, text_lengths = dp.GetTextData(data)
    test_data, test_lengths, test_matches = dp.GetTestData(data)
    print('Getting evaluation data ready takes %f secs' % (time.time() - start))
    #
    # conf = biLSTM2M.Config(params[0], params[1], params[2], params[3], params[4], 1.0)
    # with tf.variable_scope('model'):
    #     m = biLSTM2M.SimpleLSTM(True, conf)
    # saver = tf.train.Saver(tf.all_variables())
    # with tf.Session() as sess:
    #     saver.restore(sess, '../Saved_Models/2biLSTM+margin/' + model_name)
    #     # biLSTM2M.eval(m, sess, test_data, test_lengths, test_matches, conf)
    #     # acous_text_eval(m, sess, test_data, test_lengths, text_data, text_lengths, test_matches, conf)
    #     cosine_distance(m, sess, test_data, test_lengths, text_data, text_lengths, test_matches, conf, data.files)

    args = dp.ArgsHandle()
    conf = biLSTM2.Config(args.obj, args.hs, args.m, args.lr, args.kp, args.bs, args.epo, args.inits)
    with tf.variable_scope('model'):
        m = biLSTM2.SimpleLSTM(False, conf)
    saver = tf.train.Saver(tf.all_variables())
    with tf.Session() as sess:
        saver.restore(sess, '../Saved_Models/2biLSTM_models/' + model_name)
        biLSTM2.eval(m, sess, test_data, test_lengths, test_matches, conf)

main()
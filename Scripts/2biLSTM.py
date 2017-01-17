import numpy as np
import tensorflow as tf
import time, sys
import samediff
from scipy.spatial.distance import pdist
import data_processing as dp
np.set_printoptions(threshold=np.inf)

class SimpleLSTM(object):
    def __init__(self, is_training, config):
        hidden_size = config.hidden_size
        view1_input_size = config.view1_input_size
        view2_input_size = config.view2_input_size
        margin = config.margin
        lr = config.learning_rate
        kp = config.keep_prob
        obj = config.objective

        # View 1 Layer1 x1
        self._input_x1 = input_x1 = tf.placeholder(tf.float32, [None, None, view1_input_size])
        self._input_x1_lengths = input_x1_lengths = tf.placeholder(tf.int32, [None])
        input_x1_lengths_64 = tf.to_int64(input_x1_lengths)

        if is_training and kp < 1:
            input_x1 = tf.nn.dropout(input_x1, keep_prob=kp)

        l2r_cell_layer1_view1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        r2l_cell_layer1_view1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)

        with tf.variable_scope('l2r_layer1_view1'):
            l2r_outputs_layer1_view1, _ = tf.nn.dynamic_rnn(l2r_cell_layer1_view1, input_x1, dtype=tf.float32,
                                                            sequence_length=input_x1_lengths)
        with tf.variable_scope('r2l_layer1_view1'):
            r2l_outputs_layer1_view1, _ = tf.nn.dynamic_rnn(r2l_cell_layer1_view1,
                                                            tf.reverse_sequence(input_x1, input_x1_lengths_64, 1),
                                                            dtype=tf.float32, sequence_length=input_x1_lengths)
        r2l_outputs_layer1_view1 = tf.reverse_sequence(r2l_outputs_layer1_view1, input_x1_lengths_64, 1)

        # View 1 Layer 2 x1
        input_x1_layer2 = tf.concat(2, [l2r_outputs_layer1_view1, r2l_outputs_layer1_view1], 'concat_layer1_view1_x1')

        if is_training and kp < 1:
            input_x1_layer2 = tf.nn.dropout(input_x1_layer2, keep_prob=kp)

        l2r_cell_layer2_view1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        r2l_cell_layer2_view1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)

        with tf.variable_scope('l2r_layer2_view1'):
            l2r_outputs_layer2_view1, _ = tf.nn.dynamic_rnn(l2r_cell_layer2_view1, input_x1_layer2, dtype=tf.float32,
                                                            sequence_length=input_x1_lengths)
        with tf.variable_scope('r2l_layer2_view1'):
            r2l_outputs_layer2_view1, _ = tf.nn.dynamic_rnn(r2l_cell_layer2_view1,
                                                            tf.reverse_sequence(input_x1_layer2, input_x1_lengths_64, 1),
                                                            dtype=tf.float32, sequence_length=input_x1_lengths)

        l2r_outputs_view1 = tf.gather(tf.reshape(tf.concat(1, l2r_outputs_layer2_view1), [-1, hidden_size]),
                                      tf.range(tf.shape(input_x1)[0]) * tf.shape(input_x1)[1] + input_x1_lengths - 1)
        r2l_outputs_view1 = tf.gather(tf.reshape(tf.concat(1, r2l_outputs_layer2_view1), [-1, hidden_size]),
                                      tf.range(tf.shape(input_x1)[0]) * tf.shape(input_x1)[1] + input_x1_lengths - 1)
        self._final_state = x1 = self.normalization(tf.concat(1, [l2r_outputs_view1, r2l_outputs_view1], 'concat_view1_x1'))

        if not is_training:
            return

        # input_x2
        if 2 in obj or 3 in obj:
            # View 1 Layer 1 x2
            self._input_x2 = input_x2 = tf.placeholder(tf.float32, [None, None, view1_input_size])
            self._input_x2_lengths = input_x2_lengths = tf.placeholder(tf.int32, [None])
            input_x2_lengths_64 = tf.to_int64(input_x2_lengths)

            if is_training and kp < 1:
                input_x2 = tf.nn.dropout(input_x2, keep_prob=kp)

            with tf.variable_scope('l2r_layer1_view1', reuse=True):
                l2r_outputs_layer1_view1, _ = tf.nn.dynamic_rnn(l2r_cell_layer1_view1, input_x2, dtype=tf.float32,
                                                                sequence_length=input_x2_lengths)
            with tf.variable_scope('r2l_layer1_view1', reuse=True):
                r2l_outputs_layer1_view1, _ = tf.nn.dynamic_rnn(r2l_cell_layer1_view1,
                                                                tf.reverse_sequence(input_x2, input_x2_lengths_64, 1),
                                                                dtype=tf.float32, sequence_length=input_x2_lengths)
            r2l_outputs_layer1_view1 = tf.reverse_sequence(r2l_outputs_layer1_view1, input_x2_lengths_64, 1)

            # View 1 Layer 2 x2
            input_x2_layer2 = tf.concat(2, [l2r_outputs_layer1_view1, r2l_outputs_layer1_view1], 'concat_layer1_view1_x2')

            if is_training and kp < 1:
                input_x2_layer2 = tf.nn.dropout(input_x2_layer2, keep_prob=kp)

            l2r_cell_layer2_view1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
            r2l_cell_layer2_view1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)

            with tf.variable_scope('l2r_layer2_view1', reuse=True):
                l2r_outputs_layer2_view1, _ = tf.nn.dynamic_rnn(l2r_cell_layer2_view1, input_x2_layer2,
                                                                dtype=tf.float32,
                                                                sequence_length=input_x2_lengths)
            with tf.variable_scope('r2l_layer2_view1', reuse=True):
                r2l_outputs_layer2_view1, _ = tf.nn.dynamic_rnn(r2l_cell_layer2_view1,
                                                                tf.reverse_sequence(input_x2_layer2,
                                                                                    input_x2_lengths_64, 1),
                                                                dtype=tf.float32, sequence_length=input_x2_lengths)

            l2r_outputs_view1 = tf.gather(tf.reshape(tf.concat(1, l2r_outputs_layer2_view1), [-1, hidden_size]),
                                          tf.range(tf.shape(input_x2)[0]) * tf.shape(input_x2)[
                                              1] + input_x2_lengths - 1)
            r2l_outputs_view1 = tf.gather(tf.reshape(tf.concat(1, r2l_outputs_layer2_view1), [-1, hidden_size]),
                                          tf.range(tf.shape(input_x2)[0]) * tf.shape(input_x2)[
                                              1] + input_x2_lengths - 1)
            x2 = self.normalization(tf.concat(1, [l2r_outputs_view1, r2l_outputs_view1], 'concat_view1_x2'))

        # View 2 Layer 1 c1
        self._input_c1 = input_c1 = tf.placeholder(tf.float32, [None, None, view2_input_size])
        self._input_c1_lengths = input_c1_lengths = tf.placeholder(tf.int32, [None])
        input_c1_lengths_64 = tf.to_int64(input_c1_lengths)

        l2r_cell_layer1_view2 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        r2l_cell_layer1_view2 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)

        with tf.variable_scope('l2r_layer1_view2'):
            l2r_outputs_layer1_view2, _ = tf.nn.dynamic_rnn(l2r_cell_layer1_view2, input_c1, dtype=tf.float32,
                                                            sequence_length=input_c1_lengths)
        with tf.variable_scope('r2l_layer1_view2'):
            r2l_outputs_layer1_view2, _ = tf.nn.dynamic_rnn(r2l_cell_layer1_view2,
                                                            tf.reverse_sequence(input_c1, input_c1_lengths_64, 1),
                                                            dtype=tf.float32, sequence_length=input_c1_lengths)
        r2l_outputs_layer1_view2 = tf.reverse_sequence(r2l_outputs_layer1_view2, input_c1_lengths_64, 1)

        # View 2 Layer 2 c1
        input_c1_layer2 = tf.concat(2, [l2r_outputs_layer1_view2, r2l_outputs_layer1_view2], 'concat_layer1_view2_c1')

        if is_training and kp < 1:
            input_c1_layer2 = tf.nn.dropout(input_c1_layer2, keep_prob=kp)

        l2r_cell_layer2_view2 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        r2l_cell_layer2_view2 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)

        with tf.variable_scope('l2r_layer2_view2'):
            l2r_outputs_layer2_view2, _ = tf.nn.dynamic_rnn(l2r_cell_layer2_view2, input_c1_layer2, dtype=tf.float32,
                                                            sequence_length=input_c1_lengths)
        with tf.variable_scope('r2l_layer2_view2'):
            r2l_outputs_layer2_view2, _ = tf.nn.dynamic_rnn(r2l_cell_layer2_view2,
                                                            tf.reverse_sequence(input_c1_layer2, input_c1_lengths_64, 1),
                                                            dtype=tf.float32, sequence_length=input_c1_lengths)

        l2r_outputs_view2 = tf.gather(tf.reshape(tf.concat(1, l2r_outputs_layer2_view2), [-1, hidden_size]),
                                      tf.range(tf.shape(input_c1)[0]) * tf.shape(input_c1)[1] + input_c1_lengths - 1)
        r2l_outputs_view2 = tf.gather(tf.reshape(tf.concat(1, r2l_outputs_layer2_view2), [-1, hidden_size]),
                                      tf.range(tf.shape(input_c1)[0]) * tf.shape(input_c1)[1] + input_c1_lengths - 1)
        self._word_state = c1 = self.normalization(tf.concat(1, [l2r_outputs_view2, r2l_outputs_view2], 'concat_view2_c1'))

        # input_c2
        if 0 in obj or 1 in obj:
            # View 2 Layer 1 c2
            self._input_c2 = input_c2 = tf.placeholder(tf.float32, [None, None, view2_input_size])
            self._input_c2_lengths = input_c2_lengths = tf.placeholder(tf.int32, [None])
            input_c2_lengths_64 = tf.to_int64(input_c2_lengths)

            with tf.variable_scope('l2r_layer1_view2', reuse=True):
                l2r_outputs_layer1_view2, _ = tf.nn.dynamic_rnn(l2r_cell_layer1_view2, input_c2, dtype=tf.float32,
                                                                sequence_length=input_c2_lengths)
            with tf.variable_scope('r2l_layer1_view2', reuse=True):
                r2l_outputs_layer1_view2, _ = tf.nn.dynamic_rnn(r2l_cell_layer1_view2,
                                                                tf.reverse_sequence(input_c2, input_c2_lengths_64, 1),
                                                                dtype=tf.float32, sequence_length=input_c2_lengths)
            r2l_outputs_layer1_view2 = tf.reverse_sequence(r2l_outputs_layer1_view2, input_c2_lengths_64, 1)

            # View 2 Layer 2 c2
            input_c2_layer2 = tf.concat(2, [l2r_outputs_layer1_view2, r2l_outputs_layer1_view2], 'concat_layer1_view2_c2')

            if is_training and kp < 1:
                input_c2_layer2 = tf.nn.dropout(input_c2_layer2, keep_prob=kp)

            with tf.variable_scope('l2r_layer2_view2', reuse=True):
                l2r_outputs_layer2_view2, _ = tf.nn.dynamic_rnn(l2r_cell_layer2_view2, input_c2_layer2, dtype=tf.float32,
                                                                sequence_length=input_c2_lengths)
            with tf.variable_scope('r2l_layer2_view2', reuse=True):
                r2l_outputs_layer2_view2, _ = tf.nn.dynamic_rnn(r2l_cell_layer2_view2,
                                                                tf.reverse_sequence(input_c2_layer2, input_c2_lengths_64, 1),
                                                                dtype=tf.float32, sequence_length=input_c2_lengths)

            l2r_outputs_view2 = tf.gather(tf.reshape(tf.concat(1, l2r_outputs_layer2_view2), [-1, hidden_size]),
                                          tf.range(tf.shape(input_c2)[0]) * tf.shape(input_c2)[1] + input_c2_lengths - 1)
            r2l_outputs_view2 = tf.gather(tf.reshape(tf.concat(1, r2l_outputs_layer2_view2), [-1, hidden_size]),
                                          tf.range(tf.shape(input_c2)[0]) * tf.shape(input_c2)[1] + input_c2_lengths - 1)
            c2 = self.normalization(tf.concat(1, [l2r_outputs_view2, r2l_outputs_view2], 'concat_view2_c2'))

        num_objectives = len(obj)
        loss = 0
        if 0 in obj:
            loss += self.contrastive_loss(margin, x1, c1, c2)
        if 1 in obj:
            loss += self.contrastive_loss(margin, c1, x1, c2)
        if 2 in obj:
            loss += self.contrastive_loss(margin, c1, x1, x2)
        if 3 in obj:
            loss += self.contrastive_loss(margin, x1, c1, x2)
        loss /= num_objectives
        self._loss = loss
        self._train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    def contrastive_loss(self, margin, x1, c1, c2):
        sim = tf.mul(x1, c1)
        sim = tf.reduce_sum(sim, 1)
        dis = tf.mul(x1, c2)
        dis = tf.reduce_sum(dis, 1)
        return tf.reduce_mean(tf.maximum(margin + dis - sim, 0))

    def normalization(self, x):
        norm = tf.sqrt(tf.reduce_sum(tf.square(x), 1, keep_dims=True) + 1e-8)
        return x / tf.tile(norm, [1, tf.shape(x)[1]])

    @property
    def input_x1(self):
        return self._input_x1

    @property
    def input_x2(self):
        return self._input_x2

    @property
    def input_c1(self):
        return self._input_c1

    @property
    def input_c2(self):
        return self._input_c2

    @property
    def loss(self):
        return self._loss

    @property
    def train_step(self):
        return self._train_step

    @property
    def input_x1_lengths(self):
        return self._input_x1_lengths

    @property
    def input_x2_lengths(self):
        return self._input_x2_lengths

    @property
    def input_c1_lengths(self):
        return self._input_c1_lengths

    @property
    def input_c2_lengths(self):
        return self._input_c2_lengths

    @property
    def final_state(self):
        return self._final_state

    @property
    def word_state(self):
        return self._word_state


def run_epoch(m, op, sess, obj, input_x1_set, input_x2_set, input_c1_set, input_c2_set, input_x1_lengths_set, input_x2_lengths_set, input_c1_lengths_set, input_c2_lengths_set):
    cost = 0
    num_mini_batches = len(input_x1_set)
    if 2 not in obj and 3 not in obj:
        for i in range(num_mini_batches):
            loss, _ = sess.run([m.loss, op], {m.input_x1: input_x1_set[i],
                                              m.input_c1: input_c1_set[i],
                                              m.input_c2: input_c2_set[i],
                                              m.input_x1_lengths: input_x1_lengths_set[i],
                                              m.input_c1_lengths: input_c1_lengths_set[i],
                                              m.input_c2_lengths: input_c2_lengths_set[i]})
            cost += loss
    elif 0 not in obj and 1 not in obj:
        for i in range(num_mini_batches):
            loss, _ = sess.run([m.loss, op], {m.input_x1: input_x1_set[i],
                                              m.input_c1: input_c1_set[i],
                                              m.input_x2: input_x2_set[i],
                                              m.input_x1_lengths: input_x1_lengths_set[i],
                                              m.input_c1_lengths: input_c1_lengths_set[i],
                                              m.input_x2_lengths: input_x2_lengths_set[i]})
            cost += loss
    else:
        for i in range(num_mini_batches):
            loss, _ = sess.run([m.loss, op], {m.input_x1: input_x1_set[i],
                                              m.input_c1: input_c1_set[i],
                                              m.input_x2: input_x2_set[i],
                                              m.input_c2: input_c2_set[i],
                                              m.input_x1_lengths: input_x1_lengths_set[i],
                                              m.input_c1_lengths: input_c1_lengths_set[i],
                                              m.input_x2_lengths: input_x2_lengths_set[i],
                                              m.input_c2_lengths: input_c2_lengths_set[i]})
            cost += loss
    return cost / num_mini_batches


def eval(m, sess, data, lengths, matches, config):
    embeddings = []
    now = 0
    while now < len(data):
        embedding = sess.run(m.final_state, {m.input_x1: data[now: now + config.eval_batch_size],
                                             m.input_x1_lengths: lengths[now: now + config.eval_batch_size]})
        embeddings.append(embedding)
        now += config.eval_batch_size
    X = np.vstack(embeddings)
    distances = pdist(X, 'cosine')
    ap, prb = samediff.average_precision(distances[matches == True], distances[matches == False])
    print "Average precision:", ap
    print "Precision-recall breakeven:", prb
    return ap


class Config(object):
    view1_input_size = 39
    view2_input_size = 26
    eval_batch_size = 500

    def __init__(self, obj, hs, m, lr, kp, bs, epo, inits):
        self.init_scale = float(inits)
        self.epoch = int(epo)
        self.batch_size = int(bs)
        self.hidden_size = int(hs)
        self.margin = float(m)
        self.learning_rate = float(lr)
        self.keep_prob = float(kp)

        self.objective = []
        for i in range(4):
            if str(i) in obj:
                self.objective.append(i)


def main():
    # parse the arguments, then pass them to configuration
    args = dp.ArgsHandle()
    conf = Config(args.obj, args.hs, args.m, args.lr, args.kp, args.bs, args.epo, args.inits)

    # define the model we are gonna load, see if we want to train from scratch
    last_epoch = int(args.lastepoch)
    continue_training = False
    if last_epoch > -1:
        continue_training = True
        model_name = dp.ModelName(args)

    # pad input_x1 and input_c1, get validation data ready
    input_x1_set, input_c1_set, input_x1_lengths_set, input_c1_lengths_set= dp.GetInputData('../Data/', conf)
    dev = np.load('../Data/dev.npz')
    train = np.load('../Data/train.npz')
    valid_data, valid_lengths, valid_matches = dp.GetTestData(dev)
    train_data, train_lengths, train_matches = dp.GetTestData(train)

    # get the list of all words seen in training set
    names = dp.GetNames()

    # initialize conputing graphs, get ready for saving models
    initializer = tf.random_uniform_initializer(-conf.init_scale, conf.init_scale)
    with tf.variable_scope('model', reuse=None, initializer=initializer):
        m = SimpleLSTM(True, conf)
    with tf.variable_scope('model', reuse=True, initializer=initializer):
        mvalid = SimpleLSTM(False, conf)
    saver = tf.train.Saver(tf.all_variables())

    # train the models
    with tf.Session() as sess:
        if continue_training:
            saver.restore(sess, '../Saved_Models/2biLSTM/' + model_name)
        else:
            sess.run(tf.initialize_all_variables())
        for i in range(last_epoch + 1, last_epoch + conf.epoch + 1):
            # initialize input_c2 and input_x2 to None
            input_c2_set = [None] * len(input_x1_set)
            input_c2_lengths_set = [None] * len(input_x1_set)
            input_x2_set = [None] * len(input_x1_set)
            input_x2_lengths_set = [None] * len(input_x1_set)

            if 0 in conf.objective or 1 in conf.objective:
                input_c2_set, input_c2_lengths_set = dp.GetUnmatchedLabel(train.files, names, conf)
            if 2 in conf.objective or 3 in conf.objective:
                input_x2_set, input_x2_lengths_set = dp.GetUnmatchedAcous(train, conf)

            output_name = dp.OutputName(args)
            fout = open('../Outputs/2biLSTM/' + output_name + '.txt', 'a+')
            avg_cost = run_epoch(m, m.train_step, sess, conf.objective, input_x1_set, input_x2_set, input_c1_set, input_c2_set, input_x1_lengths_set, input_x2_lengths_set, input_c1_lengths_set, input_c2_lengths_set)

            # write the results to file
            fout.write('Epoch ' + str(i) + ': ' + str(avg_cost) + '\n')

            # calculate AP every 5 epochs
            if (i+1) % 5 == 0 or i == 0:
                dev_ap = eval(mvalid, sess, valid_data, valid_lengths, valid_matches, conf)
                fout.write('Dev AP: ' + str(dev_ap) + '\n')
                train_ap = eval(mvalid, sess, train_data, train_lengths, train_matches, conf)
                fout.write('Train AP: ' + str(train_ap) + '\n')
            fout.close()
            if (i + 1) % 5 == 0 or i == 0:
                best_AP, best_idx = dp.BestAP('../Outputs/2biLSTM/' + output_name + '.txt')
                if i == last_epoch + conf.epoch or dev_ap == best_AP:
                    saver.save(sess, '../Saved_Models/2biLSTM/' + output_name, i)
                keep_indices = [best_idx, last_epoch + conf.epoch]
                dp.ModelClean(keep_indices, '../Saved_Models/2biLSTM/', output_name)

if __name__ == '__main__':
    main()
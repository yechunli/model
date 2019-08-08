import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np

def read_data(str='FNN'):
    data_dir = 'F:\python_project\mnist'
    mnist = read_data_sets(data_dir, one_hot=True)
    train_data = mnist.train.images
    train_label = mnist.train.labels
    #train_data, train_label = shuffle(train_data, train_label, random_state=0)
    test_data = mnist.test.images
    test_label = mnist.test.labels
    if str == 'RNN' or str == 'CNN':
        train_data = np.array([np.reshape(x, newshape=[28,28]) for x in mnist.train.images])
        test_data = np.array([np.reshape(x, newshape=[28,28]) for x in mnist.test.images])
        if str == 'CNN':
            train_data = np.expand_dims(train_data, -1)
            test_data = np.expand_dims(test_data, -1)
    return train_data, train_label, test_data, test_label

class FNN():
    def __init__(self, num_hidden_layers, act, input_dim, output_dim, keep_rate, learning_rate, batch_size, l2_lambda):
        self.num_hidden_layers = num_hidden_layers
        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.keep_rate = keep_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.cell_num = 500

        self.data = tf.placeholder(dtype=tf.float32, shape=input_dim)
        self.label = tf.placeholder(dtype=tf.float32, shape=output_dim)
        self.model = tf.placeholder(dtype=tf.string)

        self.create_mode()

    def initialization(self):
        weight_first = tf.Variable(tf.truncated_normal(shape=[self.input_dim[1], self.cell_num], dtype=tf.float32, stddev=0.1))
        #bias_first = tf.Variable(tf.zeros(shape=[1, self.cell_num], dtype=tf.float32))
        bias_first = tf.Variable(tf.constant(0.1, shape=[1, self.cell_num], dtype=tf.float32))
        weight_hidden = []
        bias_hidden = []
        for i in range(self.num_hidden_layers):
            weight_tmp = tf.Variable(tf.truncated_normal(shape=[self.cell_num, self.cell_num], dtype=tf.float32, stddev=0.1))
            bias_tmp = tf.Variable(tf.zeros(shape=[1, self.cell_num], dtype=tf.float32))
            weight_hidden.append(weight_tmp)
            bias_hidden.append(bias_tmp)
        weight_output = tf.Variable(tf.truncated_normal(shape=[self.cell_num, 10], dtype=tf.float32, stddev=0.1))
        #bias_output = tf.Variable(tf.zeros(shape=[1], dtype=tf.float32))
        bias_output = tf.Variable(tf.constant(0.1, shape=[10], dtype=tf.float32))
        return weight_first, bias_first, weight_hidden, bias_hidden, weight_output, bias_output

    def cal_accuracy(self, act):
        result = tf.argmax(act, axis=1)
        label = tf.argmax(self.label, axis=1)
        bool_compare = tf.equal(result, label)
        float_compare = tf.cast(bool_compare, tf.float32)
        self.accuracy = tf.reduce_mean(float_compare)

    def create_mode(self):

        weight_first, bias_first, weight_hidden, bias_hidden, weight_output, bias_output = self.initialization()
        mul = tf.matmul(self.data, weight_first)
        add = tf.add(mul, bias_first)
        act = self.act(add)
        l2_loss = tf.nn.l2_loss(weight_first)
        if self.model == 'train':
            act = tf.nn.dropout(act, keep_prob=self.keep_rate)
        for i in range(self.num_hidden_layers):
            mul = tf.matmul(act, weight_hidden[i])
            add = tf.add(mul, bias_hidden[i])
            act = self.act(add)
            l2_loss = l2_loss + tf.nn.l2_loss(weight_hidden[i])
            if self.model == 'train':
                act = tf.nn.dropout(act, keep_prob=self.keep_rate)
        mul = tf.matmul(act, weight_output)
        add = tf.add(mul, bias_output)
        l2_loss = l2_loss + tf.nn.l2_loss(weight_output)

        self.loss = loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=add))
        self.cross_loss = loss + self.l2_lambda * l2_loss
        with tf.name_scope('Loss'):
            tf.summary.scalar('loss', self.cross_loss)

        #self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_loss)

        act = tf.nn.softmax(add)
        self.cal_accuracy(act)

class CNN():
    def __init__(self, num_cnn_hidden_layers, num_dense_hidden_layers, cnn_act, dense_act, keep_rate, learning_rate, batch_size, l2_lambda):
        self.num_cnn_hidden_layers = num_cnn_hidden_layers
        self.num_dense_hidden_layers = num_dense_hidden_layers
        self.cnn_act = cnn_act
        self.dense_act = dense_act
        self.keep_rate = keep_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda

        self.data = tf.placeholder(dtype=tf.float32, shape=[batch_size, 28, 28, 1])
        self.label = tf.placeholder(dtype=tf.float32, shape=[batch_size, 10])
        self.model = tf.placeholder(dtype=tf.string)

        self.create_model()

    def cnn_initialization(self):
        weight_hidden = []
        bias_hidden = []
        weight_first = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 25], stddev=0.1))
        bias_first = tf.Variable(tf.constant(0.1, shape=[25]))
        for i in range(self.num_cnn_hidden_layers):
            tmp_weight = tf.Variable(tf.truncated_normal(shape=[3, 3, 25, 25], stddev=0.1))
            tmp_bias = tf.Variable(tf.constant(0.1, shape=[25]))
            weight_hidden.append(tmp_weight)
            bias_hidden.append(tmp_bias)
        weight_output = tf.Variable(tf.truncated_normal(shape=[3, 3, 25, 50], stddev=0.1))
        bias_output = tf.Variable(tf.constant(0.1, shape=[50]))
        return weight_first, bias_first, weight_hidden, bias_hidden, weight_output, bias_output

    def dense_initialization(self, shape):
        weight_hidden = []
        bias_hidden = []
        weight_first = tf.Variable(tf.truncated_normal(shape=[shape, 100], stddev=0.1))
        bias_first = tf.Variable(tf.constant(0.1, shape=[100]))
        for i in range(self.num_dense_hidden_layers):
            tmp_weight = tf.Variable(tf.truncated_normal(shape=[100, 100], stddev=0.1))
            tmp_bias = tf.Variable(tf.constant(0.1, shape=[100]))
            weight_hidden.append(tmp_weight)
            bias_hidden.append(tmp_bias)
        weight_output = tf.Variable(tf.truncated_normal(shape=[100, 10], stddev=0.1))
        bias_output = tf.Variable(tf.constant(0.1, shape=[10]))
        return weight_first, bias_first, weight_hidden, bias_hidden, weight_output, bias_output

    def CNN_layers(self):
        cnn_weight_first, cnn_bias_first, cnn_weight_hidden, cnn_bias_hidden, cnn_weight_output, cnn_bias_output = self.cnn_initialization()
        conv = tf.nn.conv2d(self.data, cnn_weight_first, strides=[1, 1, 1, 1], padding='SAME')
        add = tf.nn.bias_add(conv, cnn_bias_first)
        act = self.cnn_act(add)
        pool = tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        if self.model == 'train':
            pool = tf.nn.dropout(pool, keep_prob=self.keep_rate)
        for i in range(self.num_cnn_hidden_layers):
            conv = tf.nn.conv2d(pool, cnn_weight_hidden[i], strides=[1, 1, 1, 1], padding='SAME')
            add = tf.nn.bias_add(conv, cnn_bias_hidden[i])
            act = self.cnn_act(add)
            pool = tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            if self.model == 'train':
                pool = tf.nn.dropout(pool, keep_prob=self.keep_rate)
        conv = tf.nn.conv2d(pool, cnn_weight_output, strides=[1, 1, 1, 1], padding='SAME')
        add = tf.nn.bias_add(conv, cnn_bias_output)
        act = self.cnn_act(add)
        pool = tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        shape = tf.shape(pool)
        dense_input = tf.reshape(pool, shape=[self.batch_size, shape[1] * shape[2] * shape[3]])
        return dense_input


    def dense_layers(self, dense_input_dim, dense_input):
        dense_weight_first, dense_bias_first, dense_weight_hidden, dense_bias_hidden, dense_weight_output, dense_bias_output = self.dense_initialization(
            shape=dense_input_dim)
        mat = tf.matmul(dense_input, dense_weight_first)
        add = tf.add(mat, dense_bias_first)
        act = self.dense_act(add)
        if self.model == 'train':
            act = tf.nn.dropout(act, keep_prob=self.keep_rate)
        for i in range(self.num_dense_hidden_layers):
            mat = tf.matmul(act, dense_weight_hidden[i])
            add = tf.add(mat, dense_bias_hidden[i])
            act = self.dense_act(add)
            if self.model == 'train':
                act = tf.nn.dropout(act, keep_prob=self.keep_rate)
        mat = tf.matmul(act, dense_weight_output)
        add = tf.add(mat, dense_bias_output)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=add, labels=self.label))
        act = tf.nn.softmax(add)
        return act, loss

    def cal_accuracy(self, act):
        result = tf.argmax(act, axis=1)
        label = tf.argmax(self.label, axis=1)
        bool_compare = tf.equal(result, label)
        float_compare = tf.cast(bool_compare, tf.float32)
        self.accuracy = tf.reduce_mean(float_compare)

    def create_model(self):
        #cnn part
        dense_input = self.CNN_layers()

        #dense part
        dense_input_dim = dense_input.get_shape().as_list()[1]
        act, self.loss = self.dense_layers(dense_input_dim, dense_input)
        with tf.name_scope('Loss'):
            tf.summary.scalar('loss', self.loss)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.cal_accuracy(act)

class RNN():
    def __init__(self, num_rnn_hidden_layers, num_dense_hidden_layers, dense_act, keep_rate, learning_rate, batch_size, l2_lambda):
        self.num_rnn_hidden_layers = num_rnn_hidden_layers
        self.num_dense_hidden_layers = num_dense_hidden_layers
        self.dense_act = dense_act
        self.keep_rate = keep_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda

        self.data = tf.placeholder(dtype=tf.float32, shape=[batch_size, 28, 28])
        self.label = tf.placeholder(dtype=tf.float32, shape=[batch_size, 10])
        self.model = tf.placeholder(dtype=tf.string)

        self.create_model()

    def dense_initialization(self):
        weight_first = tf.Variable(tf.truncated_normal(shape=[50, 50], stddev=0.1))
        bias_first = tf.Variable(tf.constant(0.1, shape=[50]))
        weight_hidden = []
        bias_hidden = []
        for i in range(self.num_dense_hidden_layers):
            weight_tmp = tf.Variable(tf.truncated_normal(shape=[50, 50], stddev=0.1))
            bias_tmp = tf.Variable(tf.constant(0.1, shape=[50]))
            weight_hidden.append(weight_tmp)
            bias_hidden.append(bias_tmp)
        weight_output = tf.Variable(tf.truncated_normal(shape=[50, 10], stddev=0.1))
        bias_output = tf.Variable(tf.constant(0.1, shape=[10]))
        return weight_first, bias_first, weight_hidden, bias_hidden, weight_output, bias_output

    def create_model(self):
        static_cell = []
        for i in range(self.num_rnn_hidden_layers+1):
            cell = tf.nn.rnn_cell.BasicLSTMCell(50, forget_bias=1)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.keep_rate)
            static_cell.append(cell)
        #cells = tf.nn.rnn_cell.MultiRNNCell((self.num_rnn_hidden_layers + 1) * [cell])
        cells = tf.nn.rnn_cell.MultiRNNCell(static_cell)
        initial_state = cells.zero_state(self.batch_size, tf.float32)

        self.output, self.state = tf.nn.dynamic_rnn(cells, self.data, initial_state=initial_state)

        weight_first, bias_first, weight_hidden, bias_hidden, weight_output, bias_output = self.dense_initialization()
        mat = tf.matmul(self.state[0][1], weight_first)
        add = tf.nn.bias_add(mat, bias_first)
        act = self.dense_act(add)
        if self.model == 'train':
            act = tf.nn.dropout(act, keep_prob=self.keep_rate)
        for i in range(self.num_dense_hidden_layers):
            mat = tf.matmul(act, weight_hidden[i])
            add = tf.nn.bias_add(mat, bias_hidden[i])
            act = self.dense_act(add)
            if self.model == 'train':
                act = tf.nn.dropout(act, keep_prob=self.keep_rate)
        mat = tf.matmul(act, weight_output)
        add = tf.nn.bias_add(mat, bias_output)
        act = tf.nn.softmax(add)

        with tf.name_scope('Loss'):
            self.loss = loss = tf.reduce_mean((act - self.label) ** 2)
            tf.summary.scalar('loss', loss)

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        result = tf.argmax(act, axis=1)
        label = tf.argmax(self.label, axis=1)
        bool_compare = tf.equal(result, label)
        accuracy = tf.cast(bool_compare, tf.float32)
        self.accuracy = tf.reduce_mean(accuracy)

type = 'RNN'
train_data, train_label, test_data, test_label = read_data(type)
epoch = 10
learning_rate = 0.001
keep_rate = 0.9
batch_size = 100
l2_lambda = 0
tf.train.GradientDescentOptimizer

#fnn part
if type == 'FNN':
    input_dim = [batch_size, np.shape(train_data)[1]]
    output_dim = [batch_size, 10]
    num_hidden_layers = 1
    act = tf.nn.relu
    mode = FNN(num_hidden_layers, act, input_dim, output_dim, keep_rate, learning_rate, batch_size, l2_lambda)
    saver_dir = 'F:\python_project\model\\fnn_model'

#cnn part
elif type == 'CNN':
    num_cnn_hidden_layers = 0
    num_dense_hidden_layers = 0
    cnn_act = tf.nn.relu
    dense_act = tf.nn.relu
    mode = CNN(num_cnn_hidden_layers, num_dense_hidden_layers, cnn_act, dense_act, keep_rate, learning_rate, batch_size, l2_lambda)
    saver_dir = 'F:\python_project\model\cnn_model'

elif type == 'RNN':
    num_rnn_hidden_layers = 1
    num_dense_hidden_layers = 0
    dense_act = tf.nn.relu
    mode = RNN(num_rnn_hidden_layers, num_dense_hidden_layers, dense_act, keep_rate, learning_rate, batch_size, l2_lambda)
    saver_dir = 'F:\python_project\model\\rnn_model'


saver = tf.train.Saver()
with tf.Session() as sess:
    if os.path.exists(saver_dir+'.index'):
        saver.restore(sess, saver_dir)
    else:
        sess.run(tf.initialize_all_variables())
    writer = tf.summary.FileWriter('F:\python_project\log\mnist', graph=sess.graph)
    merge = tf.summary.merge_all()
    iterator_train = int(np.shape(train_data)[0] / batch_size - 1)
    iterator_test = int(np.shape(test_data)[0] / batch_size - 1)
    k = 0
    for i in range(epoch):
        for j in range(iterator_train):
            feed_train = {mode.data: train_data[j * batch_size: (j + 1) * batch_size],
                    mode.label: train_label[j * batch_size: (j + 1) * batch_size],
                    mode.model: 'train'}
            _, loss = sess.run([mode.train_op, mode.loss], feed_dict=feed_train)
            #print('train loss %.2f, accuracy %.2f' % (loss, accuracy))
            if j % 10 == 0:
                print('train loss %.2f' % loss)
                feed_test = {mode.data: test_data[k * batch_size: (k + 1) * batch_size],
                        mode.label: test_label[k * batch_size: (k + 1) * batch_size],
                        mode.model: 'test'}
                test_accuracy, summary= sess.run([mode.accuracy, merge], feed_dict=feed_test)
                print('test accuracy %.2f' % test_accuracy)
                k = k + 1
                if k == iterator_test:
                    print('one test end, another started')
                    k = 0
                writer.add_summary(summary, int(j%10))
    saver.save(sess, saver_dir)
    writer.close()




#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'grasses'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__   = 'Copyright © 2017/09/30, grasses'

import tensorflow as tf, numpy as np, os, errno, math
from util.modeldb import ModelDB
from util.board import BoardUtil

from sklearn.metrics import confusion_matrix

curr_path = os.path.dirname(os.path.realpath(__file__))

class Network(object):
    def __init__(self,  conf):
        self.conf = conf["net"]
        self.test_conf = conf["test"]
        self.train_conf = conf["train"]

        # vote
        self.vote_base = self.test_conf["vote"]["base"]
        self.mutiple_view = self.test_conf["vote"]["mutiple_view"]
        self.mutiple_base = self.test_conf["vote"]["mutiple_base"]
        self.mutiple_label = self.test_conf["vote"]["mutiple_label"]
        self.with_mutiple_vote = True if self.test_conf["vote"]["type"] == "mutiple" else False

        self.dropout_rate = self.conf["dropout_rate"]
        self.optimizer_method = self.conf["optimizer_method"]
        self.learning_rate = self.conf["learning_rate"]
        self.decay_rate = self.conf["decay_rate"]
        self.decay_steps = self.conf["decay_steps"]
        self.total_batch = self.conf["total_batch"]
        self.label_count = self.conf["label_count"]
        self.image_size = self.conf["image_size"]
        self.scope_name = self.conf["scope_name"]
        self.board_steps = self.test_conf["board_steps"]

        # summary
        #self.optimizer = []
        self.merged = None
        #self.loss = []
        self.train_summaries = []
        self.test_summaries = []
        self.merged_train_summary = []
        self.writer = None
        self.writer_path = "{:s}/board/{:s}".format(curr_path, self.scope_name)

        # run
        self.train_batch_size = self.conf["batch_size"]
        self.test_batch_size = self.conf["batch_size"]

        # data
        self.train_prediction = 0

        # Graph Related
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.accuracy = None

        # save 保存训练的模型
        self.saver = None
        self.save_path = "{:s}/model/{:s}/model.ckpt".format(curr_path, self.scope_name)

        # Hyper Parameters
        self.conv_config = []  # list of dict
        self.fc_config = []  # list of dict
        self.conv_weights = []
        self.conv_biases = []
        self.conv_strides = []
        self.fc_weights = []
        self.fc_biases = []

        # svm
        self.use_svm = False
        self.svm_config = {
            "delta": None,
            "weights": None,
            "biases": None,
            "regulation_rate": 5e-4
        }

        # dnn
        self.use_dnn = False
        self.dnn_weights = []
        self.dnn_biases = []
        self.dnn_config = []

        # db
        self.ModelDB = ModelDB(fpath="{:s}/model/{:s}".format(curr_path, self.scope_name))
        self.db = self.ModelDB.read()
        self.classes = self.ModelDB.get_labels()
        # board
        self.BoardUtil = BoardUtil(conf=conf, fpath="{:s}/board/{:s}".format(curr_path, self.scope_name))
        # init function
        self.rebuild()

    def rebuild(self):
        model_path = "{:s}/model/{:s}/".format(curr_path, self.scope_name)
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except OSError as e:
            print("=> [error] build_directory() e={:s}".format(e))
            if e.errno == errno.EEXIST and os.path.isdir(model_path):
                pass
            else:
                raise
    '''
    define input placeholder
    '''
    def define_inputs(self):
        with tf.name_scope("inputs"):
            train_X_shape = test_X_shape = [self.conf["batch_size"], self.conf["image_size"], self.conf["image_size"], self.conf["image_chanel"]]
            train_y_shape = [self.conf["batch_size"], self.conf["label_count"]]

            self.train_X = tf.placeholder(tf.float32, shape=train_X_shape, name="train_X_shape")
            self.train_y = tf.placeholder(tf.float32, shape=train_y_shape, name="train_y_shape")
            self.test_X = tf.placeholder(tf.float32, shape=test_X_shape, name="test_X_shape")
            print("=> define_inputs() train_X={0}, tarin_Y={1}, test_X={2}".format(train_X_shape, train_y_shape, test_X_shape))

    def visualize_filter_map(self, tensor, how_many, display_size, name):
        filter_map = tensor[-1]
        filter_map = tf.transpose(filter_map, perm=[2, 0, 1])
        filter_map = tf.reshape(filter_map, (how_many, display_size, display_size, 1))
        self.test_summaries.append(tf.summary.image(name, tensor=filter_map, max_outputs=how_many))

    def apply_regularization(self, _lambda):
        # L2 regularization for the fully connected parameters
        regularization = 0.0
        for weights, biases in zip(self.fc_weights, self.fc_biases):
            regularization += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
        # 1e5
        return _lambda * regularization

    def print_confusion_matrix(self, confusionMatrix):
        print("Confusion    Matrix:")
        for i, line in enumerate(confusionMatrix):
            print(line, line[i] / np.sum(line))
        a = 0
        for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
            a += (column[i] / np.sum(column)) * (np.sum(column) / 26000)
            print(column[i] / np.sum(column),)
        print("\n", np.sum(confusionMatrix), a)

    def normalize(self, samples):
        return samples / 128.0 - 1.0

    def reformat(self, labels):
        one_hot_labels = []
        for num in labels:
            one_hot = [0.0] * self.label_count
            if num == self.label_count:
                one_hot[0] = 1.0
            else:
                one_hot[num] = 1.0
            one_hot_labels.append(one_hot)
        labels = np.array(one_hot_labels).astype(np.float32)
        return labels

    '''
    add convolution layer
    '''
    def add_conv(self, name, patch_size, in_depth, out_depth, stride, activation='relu', pooling=False, stddev=0.1):
        self.conv_config.append({
            "name": name,
            "patch_size": patch_size,
            "in_depth": in_depth,
            "out_depth": out_depth,
            "activation": activation,
            "pooling": pooling,
            "stride": stride,
        })

        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, in_depth, out_depth], stddev=stddev), name=name + "_weights")
            biases = tf.Variable(tf.constant(0.1, shape=[out_depth]), name=name + "_biases")
            self.conv_weights.append(weights)
            self.conv_biases.append(biases)
            self.conv_strides.append(stride)

    '''
    add full connection
    '''
    def add_fc(self, name, in_num_nodes, out_num_nodes, activation="relu", stddev=0.1):
        self.fc_config.append({
            "name": name,
            "in_num_nodes": in_num_nodes,
            "out_num_nodes": out_num_nodes,
            "activation": activation,
        })

        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([in_num_nodes, out_num_nodes], stddev=stddev))
            biases = tf.Variable(tf.constant(0.1, shape=[out_num_nodes], name=name + "_biases"))
            self.fc_weights.append(weights)
            self.fc_biases.append(biases)

            self.train_summaries.append(tf.summary.histogram("cnn_fc_" + str(len(self.fc_weights)) + "_weights", weights))
            self.train_summaries.append(tf.summary.histogram("cnn_fc_" + str(len(self.fc_biases)) + "_biases", biases))
    '''
    add svm layer
    '''
    def add_svm(self, name, in_num_nodes, out_num_nodes, stddev=0.1):
        self.use_svm = True
        with tf.name_scope(name):
            self.svm_config["weights"] = tf.Variable(tf.truncated_normal([in_num_nodes, out_num_nodes], stddev=stddev))
            self.svm_config["biases"] = tf.Variable(tf.zeros([out_num_nodes]))
            self.svm_config["delta"] = tf.constant(1.0)
    '''
    add deep nerual network
    '''
    def add_dnn(self, name, in_num_nodes, out_num_nodes, activation='relu', stddev=0.1):
        self.use_dnn = True
        self.dnn_config.append({
            "name": name,
            "in_num_nodes": in_num_nodes,
            "out_num_nodes": out_num_nodes,
            "activation": activation
        })

        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([in_num_nodes, out_num_nodes], stddev=stddev))
            biases = tf.Variable(tf.constant(0.1, shape=[out_num_nodes], name=name + '_biases'))
            self.dnn_weights.append(weights)
            self.dnn_biases.append(biases)

            self.train_summaries.append(tf.summary.histogram('dnn_' + str(len(self.fc_weights)) + '_weights', weights))
            self.train_summaries.append(tf.summary.histogram('dnn_' + str(len(self.fc_biases)) + '_biases', biases))

    '''
    define graph && model
    '''
    def define_model(self, is_train=True):
        name_scope = "train" if is_train else "test"
        def model(data_flow, is_train=True, seed=123456):
            # convolutional layers
            print("=> define_model() data_flow.shape={0}".format(data_flow.get_shape()))
            for i, (weight, bias, stride, config) in enumerate(zip(self.conv_weights, self.conv_biases, self.conv_strides, self.conv_config)):
                with tf.name_scope(config["name"] + "_model"):
                    with tf.name_scope("convolution"):
                        data_flow = tf.nn.conv2d(data_flow, filter=weight, strides=stride, padding="SAME") + bias
                        print("=> define_model() cnn_{:d}: data_flow.shape={:s}".format(i + 1, data_flow.shape))
                        '''
                        if not is_train:
                            self.visualize_filter_map(data_flow, how_many=config['out_depth'], display_size=self.image_size // (i // 2 + 1), name=config['name'] + '_conv')
                        '''
                    if config["activation"] == "relu":
                        data_flow = tf.nn.relu(data_flow)
                        '''
                        if not is_train:
                            self.visualize_filter_map(data_flow, how_many=config['out_depth'], display_size=32 // (i // 2 + 1), name=config['name'] + '_relu')
                        '''
                    if config["pooling"]:
                        data_flow = tf.nn.max_pool(
                            data_flow,
                            ksize=self.conf["pooling"]["ksize"],
                            strides=self.conf["pooling"]["strides"],
                            padding=self.conf["pooling"]["padding"]
                        )
                        '''
                        if not is_train:
                            self.visualize_filter_map(data_flow,how_many=config['out_depth'], display_size=32 // (i // 2 + 1) // 2, name=config['name'] + '_pooling')
                        '''

            # full connect layer
            for i, (weight, bias, config) in enumerate(zip(self.fc_weights, self.fc_biases, self.fc_config)):
                if i == 0:
                    shape = data_flow.get_shape().as_list()
                    data_flow = tf.reshape(data_flow, [-1, shape[1] * shape[2] * shape[3]])

                with tf.name_scope(config["name"] + "_model"):
                    with tf.name_scope("full_connection"):
                        # dropout
                        if is_train and i == len(self.fc_config) - 1:
                            data_flow = tf.nn.dropout(data_flow, self.dropout_rate, seed=seed)
                        data_flow = tf.matmul(data_flow, weight) + bias
                        if config["activation"] == "relu":
                            data_flow = tf.nn.relu(data_flow)
                        elif config["activation"] == "sigmoid":
                            data_flow = tf.nn.sigmoid(data_flow)
                        elif config["activation"] is "tanh":
                            data_flow = tf.nn.tanh(data_flow)
                        elif config["activation"] is None:
                            pass
                        else:
                            raise Exception("Activation Func can only be Relu or None right now. You passed", config["activation"])
                        print("=> define_model() full_connection_{:d}: data_flow.shape={:s}".format(i + 1, data_flow.shape))
            # dnn
            for i, (weight, bias, config) in enumerate(zip(self.dnn_weights, self.dnn_biases, self.dnn_config)):
                with tf.name_scope(config["name"] + "_model"):
                    with tf.name_scope("dnn"):
                        data_flow = tf.matmul(data_flow, weight) + bias
                        if config["activation"] == "relu":
                            data_flow = tf.nn.relu(data_flow)
                        elif config["activation"] == "sigmoid":
                            data_flow = tf.nn.sigmoid(data_flow)
                        elif config["activation"] == "tanh":
                            data_flow = tf.nn.tanh(data_flow)
                        else:
                            pass
                        print("=> define_model() dnn_{:d}: data_flow.shape={:s}".format(i, data_flow.shape))
            if self.use_svm:
                data_flow = tf.matmul(data_flow, self.svm_config["weights"]) + self.svm_config["biases"]
                print("=> define_model() svm: data_flow.shape={:s}".format(data_flow.shape))
            return data_flow

        # logits
        logits = model(self.train_X, is_train)
        with tf.name_scope(name_scope):
            if self.use_svm:
                print("=> define_model() use svm")
                y = tf.reduce_sum(logits * self.train_y, 1, keep_dims=True)
                self.loss = tf.reduce_mean(tf.reduce_sum(tf.maximum(0.0, logits - y + self.svm_config["delta"]), 1)) - self.svm_config["delta"]
                self.loss += self.svm_config["regulation_rate"] * tf.nn.l2_loss(self.svm_config["weights"])
            else:
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.train_y))
                self.loss += self.apply_regularization(_lambda=5e-4)

            # prediction
            self.train_prediction = tf.nn.softmax(logits=logits, name="train_prediction")
            tf.add_to_collection("prediction", self.train_prediction)
            # accuracy
            accuracy_op = tf.equal(tf.argmax(self.train_prediction, axis=1), tf.argmax(self.train_y, axis=1))
            self.accuracy = 100 * tf.reduce_mean(tf.cast(accuracy_op, tf.float32))

            # add summary
            self.train_summaries.append(tf.summary.scalar("accuracy", self.accuracy))
            self.test_summaries.append(tf.summary.histogram("accuracy", self.accuracy))

            self.train_summaries.append(tf.summary.scalar("loss", self.loss))
            self.test_summaries.append(tf.summary.histogram("loss", self.loss))

        # learning rate
        global_step = tf.Variable(0, trainable=False, name="global_step")
        learning_rate = tf.train.exponential_decay(
            learning_rate=self.learning_rate,
            global_step=global_step,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True
        )
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

        # optimizer
        with tf.name_scope("optimizer"):
            if self.optimizer_method == "gradient":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=global_step)
            elif self.optimizer_method == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.5).minimize(self.loss, global_step=global_step)
            elif self.optimizer_method== "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=self.loss, global_step=global_step)

        # merged summary
        self.merged_train_summary = tf.summary.merge(self.train_summaries)
        self.merged_test_summary = tf.summary.merge(self.test_summaries)
        # save graph
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, X, Y, name):
        self.define_model()
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 1
        with tf.Session(config=config, graph=tf.get_default_graph()) as session:
            self.writer = tf.summary.FileWriter(self.writer_path, session.graph)

            if not os.path.exists(self.save_path + ".meta"):
                session.run(tf.global_variables_initializer())
            else:
                self.saver.restore(session, self.save_path)

            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=session, coord=coord)
            global_step = session.graph.get_tensor_by_name("global_step:0")
            for index in range(global_step.eval(), self.total_batch):
                (samples, labels, source_name) = session.run([X, Y, name])
                #_X = np.reshape(G.normalize(samples), [self.train_batch_size, self.image_size * self.image_size * 3])
                (_X, _Y) = (self.normalize(samples), self.reformat(labels))
                (_, loss, predictions, accuracy, summary) = session.run(
                    [self.optimizer, self.loss, self.train_prediction, self.accuracy, self.merged_train_summary],
                    feed_dict={
                        self.train_X: _X,
                        self.train_y: _Y
                    }
                )
                if index % self.train_conf["summary_steps"] == 0:
                    self.writer.add_summary(summary, index)
                if index % self.train_conf["saving_steps"] == 0:
                    print("=> step={:d}, loss={:.5f}, accuracy:{:.3f}%".format(index, loss, accuracy))
                    self.saver.save(session, self.save_path)


    '''
    test
    :param  iterator    function:   yield(index, (G_X, G_Y, G_name, G_offset_x, G_offset_y))
    '''
    def test(self, iterator):
        self.define_model(is_train=False)
        print("=> test() before session")
        iterator_type = self.conf["iterator_type"]
        with tf.Session(config=tf.ConfigProto(log_device_placement=False), graph=tf.get_default_graph()) as session:
            self.writer = tf.summary.FileWriter(self.writer_path, session.graph)
            self.saver.restore(session, self.save_path)

            sess = session if iterator_type == "tfrecords" else None
            label_count = self.conf["label_count"]
            vote_base = self.test_conf["vote_base"]

            (accuracies, confusion_matrices) = ([], np.zeros([label_count, label_count], dtype=np.int32)) # np.zeros([, self.conf["label_count"], self.conf["label_count"]], dtype=np.int32)
            for (index, (G_X, G_Y, G_name, G_offset_x, G_offset_y)) in iterator(session=sess):
                (_X, _Y) = (self.normalize(G_X), self.reformat(G_Y))
                (loss, predictions, summary) = session.run(
                    [self.loss, self.train_prediction, self.merged_test_summary],
                    feed_dict={
                        self.train_X: _X,
                        self.train_y: _Y
                    }
                )
                if vote_base > 0 and iterator_type == "file":
                    predictions = self.run_vote(predictions, vote_base=vote_base)

                (ac, cm) = self.run_accuracy(self.run_vote(predictions, vote_base=vote_base), _Y, need_confusion_matrix=True)
                accuracies.append(ac)
                confusion_matrices += cm

                print("=> step={:d}, loss={:.5f}, accuracy:{:.2f}%".format(index, loss, ac))
                if index % self.train_conf["summary_steps"] == 0:
                    self.writer.add_summary(summary, index)
                if (index > self.board_steps) and (index % self.board_steps == 0):
                    self.BoardUtil.confusion_matrix(matrix=confusion_matrices, classes=self.classes, steps=index)

            self.BoardUtil.confusion_matrix(matrix=confusion_matrices, classes=self.classes, steps=index)
        return(accuracies, confusion_matrices)

    def test_vote(self, iterator):
        if not self.with_mutiple_vote: return

        self.define_model(is_train=False)
        print("=> test_vote() before session")
        with tf.Session(config=tf.ConfigProto(log_device_placement=False), graph=tf.get_default_graph()) as session:            
            self.writer = tf.summary.FileWriter(self.writer_path, session.graph)
            self.saver.restore(session, self.save_path)
            print(self.mutiple_base)
            label_count = self.conf["label_count"]
            (accuracies, confusion_matrices) = ([], np.zeros([len(self.mutiple_base), label_count, label_count], dtype=np.int32))

            for (step, (G_X, G_Y, G_name, G_offset_x, G_offset_y)) in iterator(session=self.mutiple_label):
                (_X, _Y) = (self.normalize(G_X), self.reformat(G_Y))
                (loss, predictions, summary) = session.run(
                    [self.loss, self.train_prediction, self.merged_test_summary],
                    feed_dict={
                        self.train_X: _X,
                        self.train_y: _Y
                    }
                )

                for (index, base) in enumerate(self.mutiple_base):
                    (ac, cm) = self.run_vote(predictions, _Y, vote_base=base, one_test=False)
                    confusion_matrices[index] += cm

                    if base in [1, 8, 16, 32]:
                        print("=> step={:d}, loss={:.5f}, accuracy:{:.2f}% vote_base={:d}".format(step, loss, ac, base))

                if step % 100 == 0:
                    for i in range(len(self.mutiple_base)):
                        print("=> index={:d}".format(self.mutiple_base[i]))
                        print(confusion_matrices[i])
                        print("\n")

            self.BoardUtil.vote_accuracy_curve(confusion_matrices, classes=self.ModelDB.get_labels_by_id(self.mutiple_label), mutiple_base=self.mutiple_base, steps=step)
            return(accuracies, confusion_matrices)

    '''
    api_test(): just test, not publish
    '''
    def api_test(self, file_path, scope_name, iterator, output_path, vote_base=256):
        self.define_model(is_train=False)
        print("=> api_test() before session")

        save_path = "{:s}/model/{:s}/train.ckpt".format(curr_path, scope_name)
        if (not os.path.exists(file_path)) or (not os.path.exists(save_path)):
            return []

        with tf.Session(config=tf.ConfigProto(log_device_placement=False), graph=tf.get_default_graph()) as session:
            self.writer = tf.summary.FileWriter(self.writer_path, session.graph)
            self.saver.restore(session, save_path)
            #tf.global_variables_initializer().run()

            for (index, (G_X, G_Y, image_name, offset_x, offset_y)) in iterator(session=file_path):
                (_X, _Y) = (self.normalize(G_X), self.reformat(G_Y))
                (loss, predictions) = session.run(
                    [self.loss, self.train_prediction],
                    feed_dict={
                        self.train_X: _X,
                        self.train_y: _Y
                    }
                )
                modelDB = ModelDB(fpath="{:s}/model/{:s}".format(curr_path, scope_name))
                exp = int(math.log(self.train_batch_size, 4)) + 1
                predict_result = []

                for _exp in range(exp):
                    vote_base = int(math.pow(4, _exp))
                    (unique, counts) = np.unique(np.array(self.run_vote(predictions, vote_base=vote_base)), return_counts=True)
                    predict = dict(zip(np.array(unique, dtype=np.int32), ((1.0 * counts) / self.train_batch_size)))

                    labels = modelDB.get_labels()
                    for (key, value) in predict.items():
                        predict[labels[key]] = value
                        del predict[key]
                    predict_result.append({"vote_base": vote_base, "predict": predict})
                print predict_result
                return predict_result

    def confusion_matrix(self, y_label, y_predict, num_classes, vote_base=0):
        cm = np.zeros([num_classes, num_classes], dtype=np.int32)
        (y_label, y_predict) = (np.array(y_label, dtype=np.int32), np.array(y_predict, dtype=np.int32))
        for i in range(len(y_label)):
            if (vote_base != 0) and (i % vote_base == 0):
                cm[int(y_label[i]), int(y_predict[i])] += 1
            elif (vote_base == 0):
                cm[int(y_label[i]), int(y_predict[i])] += 1
        return cm

    def run_vote(self, predicts, labels, vote_base=0, one_test=False):
        vote_base = self.train_batch_size if vote_base == 0 else vote_base
        _labels = np.argmax(labels, axis=1) if len(labels.shape) > 1 else labels

        predicts = np.argmax(predicts, 1)
        _predicts = np.zeros([self.train_batch_size])
        '''
        for count in range(self.train_batch_size / vote_base):
            (u, indices) = np.unique(predicts[count * vote_base: (count + 1) * vote_base], return_inverse=True)
            max_index = u[np.argmax(np.bincount(indices))]
            _predicts[count * vote_base: (count + 1) * vote_base] = np.repeat(max_index, vote_base)
        '''
	count = 0 
	(u, indices) = np.unique(predicts[count * vote_base: (count + 1) * vote_base], return_inverse=True)
	max_index = u[np.argmax(np.bincount(indices))]
	_predicts[count * vote_base: (count + 1) * vote_base] = np.repeat(max_index, vote_base)
        _predicts[:]=_predicts[0]
        if one_test:
            cm = np.zeros([self.conf["label_count"], self.conf["label_count"]], dtype=np.int32)
            cm[int(_labels[0])][int(_predicts[0])] = 1
        else:
            cm = self.confusion_matrix(_labels, _predicts, num_classes=self.conf["label_count"], vote_base=vote_base)
        accuracy = (100.0 * np.sum(_predicts == _labels) / _predicts.shape[0])
        return (accuracy, cm)

    def run_accuracy(self, predicts, labels, need_confusion_matrix=False, vote_base=0):
        _predicts = np.argmax(predicts, axis=1) if len(predicts.shape) > 1 else predicts
        _labels = np.argmax(labels, axis=1) if len(labels.shape) > 1 else labels

        cm = self.confusion_matrix(_labels, _predicts, num_classes=self.conf["label_count"], vote_base=vote_base) if need_confusion_matrix else None
        accuracy = (100.0 * np.sum(_predicts == _labels) / _predicts.shape[0])
        return(accuracy, cm)

# 1,2,3,4,5,6,7,8 => 8
# x % 8 == 0

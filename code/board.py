#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'grasses'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__   = 'Copyright © 2017/11/02, grasses'

import itertools, os, errno, numpy as np, matplotlib.pyplot as plt

class BoardUtil(object):
    def __init__(self, conf, fpath, db=None):
        #self.db = db
        self.conf = conf["board"]
        self.test_conf = conf["test"]
        self.process_conf = conf["process"]
        self.model_conf = conf["model"]
        self.batch_size = conf["net"]["batch_size"]

        self.fpath = "{:s}/logs/".format(fpath)
        self.label_count = conf["net"]["label_count"]
        self.scope_name = conf["net"]["scope_name"]
        self.vote_base = conf["test"]["vote_base"]
        self.with_mutiple_vote = True if self.test_conf["mutiple_vote"] == "True" else False
        self.with_normalize = True if conf["board"]["confustion_matrix"]["with_normalize"] == "True" else False

        self.compress_rate = 1.0 * conf["process"]["compress"]["quality"] / 100

        # rebuild
        self.rebuild()

    def rebuild(self):
        try:
            if not os.path.exists(self.fpath):
                os.makedirs(self.fpath)
        except OSError as e:
            print("=> [error] build_directory() e={:s}".format(e))
            if e.errno == errno.EEXIST and os.path.isdir(self.fpath):
                pass
            else:
                raise

    def gen_into(self, describe, steps, base=0):
        title = "{:s}(vote_base={:d})".format(describe, base)
        info = "scope={:s} vote_base={:d}".format(self.scope_name, base)
        name = "{:s}/{:s}_vote.{:d}_steps.{:d}.jpg".format(self.fpath, describe, base, steps)

        if self.process_conf["type"] != "None":
            info = "{:s} process={:s}_quality={:d}".format(info, self.process_conf["type"], self.process_conf["quality"])
        return (title, name, info)

    def save_confusion_matrix(self, title, name, info, matrix, classes, with_show=False, cmap=plt.cm.Blues):
        if self.with_normalize: matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        plt.switch_backend('agg')
        plt.figure(figsize=(self.label_count * 0.2 + 8, self.label_count * 0.2 + 6))
        plt.imshow(matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        plt.figtext(0.01, 0.05, info, color='black')

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=40)
        plt.yticks(tick_marks, classes)

        fmt = '.3f' if self.with_normalize else 'd'
        thresh = matrix.max() / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, format(matrix[i, j], fmt) if matrix[i, j] != 0 else "0.00", horizontalalignment="center", color="white" if matrix[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(name)
        if with_show: plt.show()

    def vote_accuracy_curve(self, matrix, classes, steps):
        (title, name, info) = self.gen_into("Voting Size & Accuracy", steps=steps, base=self.vote_base)

        plt.switch_backend('agg')
        plt.figure()
        plt.title(title)
        x = np.arange(0, self.batch_size / 2 + 1, self.vote_base, dtype=np.int32)

        total_sum = sum(matrix[:][0][0])
        for index in range(self.label_count):
            plt.plot(x, np.array(matrix.T[index][index], dtype=np.float32) / total_sum, label="{:s}".format(classes[index]))

        plt.legend()
        plt.xlabel("Voting size")
        plt.ylabel("Accuracy / (%)")
        plt.savefig(name)
        #plt.show()

    def confusion_matrix(self, matrix, classes, steps=0, with_show=False):
        (title, name, info) = self.gen_into(describe="Confusion matrix", steps=steps, base=self.vote_base)

        if self.with_mutiple_vote:
            counter = 0
            for base in np.arange(0, self.batch_size / 2 + 1, self.vote_base, dtype=np.int32):
                (title, name, info) = self.gen_into(describe="Confusion matrix", steps=steps, base=base)
                self.save_confusion_matrix(title, name, info, matrix[counter], classes, with_show)
                counter += 1
        else:
            self.save_confusion_matrix(title, name, info, matrix, classes, with_show)
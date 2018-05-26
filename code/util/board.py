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

        self.plot_type = ["confustion", "voting"]

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

    def vote_accuracy_curve(self, matrix, classes, mutiple_base, steps):
        (title, name) = ("Accuracy with Voting pactach", "{:s}/{:s}".format(self.fpath, "accuracy_voting.png"))

        plt.switch_backend('agg')
        plt.figure()
        plt.title(title)

        _matrix = np.zeros(matrix.shape, dtype=np.float32)

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                tmp_sum = np.sum(matrix[:][i][j])
                for k in range(len(matrix[i][j])):
                    _matrix[i][j][k] = 100.0 * matrix[i][j][k] / tmp_sum
        
        tmp = np.zeros(len(matrix), dtype=np.float32)
        for i in range(len(_matrix)):
            print("=> index={:d}".format(i))
            for j in range(len(_matrix[i])):
                print(_matrix[i][j])
                if (j != 8) and (j != 9): 
                    tmp[i] += _matrix[i][j][j]
            tmp[i] = tmp[i] / 18.0
            print("\n")
            
        print("\n\n\n")
        print(tmp)

        for index in range(len(classes)):
            plt.plot(mutiple_base, _matrix.T[index][index], "x-", markersize=4, label="Our method(64x64 patches)")

        x = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
             30, 31, 32]
        ], dtype=np.float)

        y = np.array([
            [97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97,
             97, 97, 97, 97, 97],
            [93.5, 93.8, 95, 95.3, 95.8, 95.9, 96, 96.1, 96.2, 96.1, 96.21, 96.24, 96.31, 96.33, 96.35, 96.39, 96.42,
             96.34, 96.36, 96.40, 96.47, 96.44, 96.45, 96.43, 96.44, 96.39, 96.44, 96.45, 96.46, 96.49, 96.50, 96.51],
            [84.8, 86, 88.8, 90.3, 90.7, 91, 91.2, 91.6, 91.8, 92, 92.12, 92.14, 92.19, 92.22, 92.23, 92.32, 92.34,
             92.36, 92.30, 92.31, 92.37, 92.40, 92.37, 92.38, 92.43, 92.45, 92.48, 92.49, 92.48, 92.42, 92.40, 92.38],
        ], dtype=np.float)

        label = [
            "Chen et al.(Whole image)",
            "Bondi et al. CNN (64x64 patches)",
            "Chen et al.(64x64 patches)"
        ]

        style = [
            ":",
            "o-",
            "*-"
        ]

        for index in range(len(label)):
            plt.plot(x[index], y[index], style[index], markersize=4, label="{:s}".format(label[index]))

        plt.legend()
        plt.xlabel("Patches per image")
        plt.ylabel("Average accuracy / [%]")
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

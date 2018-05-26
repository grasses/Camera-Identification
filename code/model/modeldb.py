#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'grasses'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__   = 'Copyright © 2017/10/31, grasses'

import json, os

class ModelDB(object):
    def __init__(self, fpath, fname="db.json"):
        self.fpath = fpath
        self.fname = fname
        self.data = None
        self.file_path = "{:s}/{:s}".format(self.fpath, self.fname)

    def get_filename(self):
        return self.file_path

    def exists(self):
        return os.path.exists(self.file_path)

    def get_labels(self):
        if self.exists():
            if self.data == None: self.read()
            models = []
            for (label, model) in self.data["label_model"].items():
                models.append(model)
            return models

    def get_labels_by_id(self, label_list):
        if self.exists():
            if self.data == None: self.read()
            models = []
            for (label, model) in self.data["label_model"].items():
                if label in label_list:
                    models.append(model)
            return models

    def write(self, data):
        with open("{:s}/{:s}".format(self.fpath, self.fname), "w") as f:
            json.dump(data, f)
            self.data = data

    def read(self):
        with open("{:s}/{:s}".format(self.fpath, self.fname), "r") as f:
            self.data = json.load(f)
            for (label, model) in self.data["label_model"].items():
                self.data["label_model"][int(label)] = model
                del self.data["label_model"][label]
            return self.data
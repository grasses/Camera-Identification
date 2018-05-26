#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'grasses'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__   = 'Copyright © 2017/11/09, grasses'

import os, threading, errno, time, json
from PIL import Image

COLOR_WHITE = "\033[1;34m{0}\033[00m"
COLOR_BLUE = "\033[1;36m{0}\033[00m"
curr_path = os.path.dirname(os.path.realpath(__file__))

class ProcessUtil(threading.Thread):
    def __init__(self, conf, fpath="", action=None):
        threading.Thread.__init__(self)
        self.fpath = fpath
        self.action = action
        self.conf = conf["process"]

        self.process_type = self.conf["type"]
        self.process_rate = self.conf[self.process_type]["quality"]
        self.process_prefix = self.conf[self.process_type]["save_prefix"]
        self.process_extension = self.conf[self.process_type]["extension"]
        self.mutile_prefix = self.conf["mutiple_prefix"]

        self.logs_path = "{:s}/../logs/process".format(curr_path)
        self.rebuild()

    def rebuild(self):
        try:
            if not os.path.exists(self.logs_path):
                os.makedirs(self.logs_path)
        except OSError as e:
            print("=> [error] rebuild() e={:s}".format(e))
            if e.errno == errno.EEXIST and os.path.isdir(self.logs_path):
                pass
            else:
                raise

    def save_logs(self, logs):
        try:
            fp = open("{:s}/{:s}_{:s}.log".format(self.logs_path, self.process_type, time.strftime("%Y-%m-%d_%H:%M", time.gmtime())), "a")
            fp.write(logs + "\n")
            fp.close()
        except Exception as e:
            print("=> [error] save_logs() e={0}".format(e))

    def get_size(self, folder):
        file_size = 0
        int_size = 0
        for (path, dirs, files) in os.walk(folder):
            file_size += len(files)
            for file in files:
                filename = os.path.join(path, file)
                int_size += os.path.getsize(filename)
        str_size = "%0.1f MB" % (int_size / (1024 * 1024.0))
        return(file_size, int_size, str_size)

    def set_fpath(self, fpath):
        self.fpath = fpath

    def set_action(self, action):
        self.action = action

    def remove(self):
        logs = "[delete]: date={:s} path={:s}".format(time.strftime("%Y-%m-%d %H:%M", time.gmtime()), self.fpath)
        for subdir, dirs, files in os.walk(self.fpath):
            for file in files:
                if file.startswith(self.mutile_prefix):
                    full_path = os.path.join(subdir, file)
                    try:
                        os.remove(full_path)
                        logs = "{:s}\n{:s}".format(logs, full_path)
                        print(COLOR_WHITE.format("Remove: \t{:s}".format(full_path)))
                    except Exception as e:
                        print("=> [error] remove() e={:s}".format(e))
        self.save_logs(logs)

    def compress(self):
        logs = "[compress]: date={:s} path={:s}".format(time.strftime("%Y-%m-%d %H:%M", time.gmtime()), self.fpath)
        (file_size, pre_int_size, pre_str_size) = self.get_size(self.fpath)
        count = 0
        for subdir, dirs, files in os.walk(self.fpath):
            for file in files:
                if file.startswith(self.process_prefix): continue
                image = os.path.join(subdir, file)
                if image.endswith(self.process_extension):
                    count = count + 1
                    im = Image.open(image)
                    im.save("{:s}/{:s}{:s}".format(self.fpath, self.process_prefix, file), quality=self.process_rate)
                    logs = "{:s}\n{:s}".format(logs, image)
                    print(COLOR_WHITE.format("Compressing:\t{:s}".format(image)))

        print(COLOR_BLUE.format("Compressed completed {:d} images".format(count)))
        (file_size, post_int_size, post_str_size) = self.get_size(self.fpath)
        print COLOR_BLUE.format("Size of folder: {:s} => {:s}".format(pre_str_size, post_str_size))
        self.save_logs(logs)

    def compress_single(self, file_path):
        full_path = "{:s}/{:s}{:s}".format(os.path.dirname(file_path), self.process_prefix, os.path.basename(file_path))
        im = Image.open(file_path)
        im.save(full_path, quality=self.process_rate)
        print(COLOR_BLUE.format("Compressed image={:s}".format(full_path)))

    def run(self):
        self.action()

def clean():
    with open("{:s}/../conf.json".format(curr_path), "r") as f:
        conf = json.load(f)

    input_path = conf["model"]["input_path"]
    for factory in os.listdir(input_path):
        factory_path = "{:s}/{:s}".format(input_path, factory)
        for model in os.listdir(factory_path):
            fpath = "{:s}/{:s}".format(factory_path, model)
            print("=> clean() fpath={:s}".format(fpath))
            ProcessUtil(conf, fpath=fpath, action=ProcessUtil.remove).start()
if __name__ == "__main__":
    clean()
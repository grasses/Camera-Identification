#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'grasses'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__   = 'Copyright © 2017/10/23, grasses'

import tensorflow as tf, numpy as np, os, random, math, errno
from scipy import misc
from util.modeldb import ModelDB
from util.process import ProcessUtil

curr_path = os.path.abspath(os.curdir)
class Generator(object):
    def __init__(self, conf, is_debug=False):
        self.is_debug = is_debug

        self.conf = conf
        self.net_conf = conf["net"]
        self.model_conf = conf["model"]
        self.process_conf = conf["process"]

        # cross validation
        self.total_batch = conf["net"]["total_batch"]
        self.cross_type = self.model_conf["cross_validation"]["type"]
        self.cross_rate = self.model_conf["cross_validation"][self.cross_type]["rate"]
        self.test_size = self.model_conf["cross_validation"]["random"]["test_size"]

        # input && output
        self.batch_size = self.net_conf["batch_size"]            # number of patchs croping from an image
        self.input_path = self.model_conf["input_path"]
        self.output_path = self.model_conf["output_path"]

        # process
        self.with_process = True if self.process_conf["type"] != "None" else False
        self.process_type = self.process_conf["type"]
        self.process_rate = self.process_conf[self.process_type]["quality"]
        self.process_prefix = self.process_conf[self.process_type]["save_prefix"]
        self.process_extension = self.process_conf[self.process_type]["extension"]

        self.scope_name = conf["net"]["scope_name"]
        self.label_count = conf["net"]["label_count"]
        self.max_image_noise = self.model_conf["max_image_noise"]  # variance of pixels
        self.generator_type = self.model_conf["generator_type"]    # generator_type = ['rand', 'slide'] => slide=slide windows; rand=rand crop
        self.min_after_dequeue = self.model_conf["min_after_dequeue"]

        self.tmp_var = {}
        self.config = {
            "model_label": {},
            "label_model": {},
            "train_list": {},
            "test_list": {},
            "label_path": {},
            "train_label_count": {},
            "test_label_count": {},
            "scope_name": self.scope_name,
            "label_count": self.label_count,

            # net
            "decay_rate": conf["net"]["decay_rate"],
            "decay_steps": conf["net"]["decay_steps"],
            "dropout_rate": conf["net"]["dropout_rate"],
            "learning_rate": conf["net"]["learning_rate"],

            # cross validation
            "cross_type": self.cross_type,
            "cross_rate": self.cross_rate,
            "test_size": self.test_size,

            # process
            "process_type": self.process_type,
            "process_rate": self.process_rate,
            "process_prefix": self.process_prefix,
            "process_extension": self.process_extension
        }
        self.ModelDB = ModelDB(fpath="{:s}/model/{:s}".format(curr_path, self.scope_name))
        self.ProcessUtil = ProcessUtil(conf=conf, action=ProcessUtil.compress)
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
        print("=> Generator() input_path={:s}, outpath={:s} ".format(self.input_path, self.output_path))

    '''
    :param value tfrecords int to init64
    '''
    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    '''
    :param value tfrecords int to float32
    '''
    def _float32_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    '''
    :param value tfrecords string to byte
    '''
    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    '''
    normalize input X matrix value from 0~255 to -1~1
    :param samples   numpy.array()
    :return samples  numpy.array()
    '''
    def normalize(self, samples):
        return samples / 128.0 - 1.0

    '''
    reformat label Y matrix, for example: from Y=2 to Y=[0, 0, 1, 0, 0], where 1 is active value and label size is 5
    :param labels    int
    :return labels   numpy.array()
    '''
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
    read directory and set attribute
    '''
    def read_directory(self):
        if self.ModelDB.exists():
            self.config = self.ModelDB.read()
            return self
        try:
            os.remove("{:s}/.DS_Store".format(self.input_path))
        except Exception:
            pass

        count = 0
        for factory in sorted(os.listdir(self.input_path)):
            tmp_path = "{:s}/{:s}".format(self.input_path, factory)
            try:
                for name in sorted(os.listdir(tmp_path)):
                    model = "{:s}_{:s}".format(factory, name)
                    model_dir = "{:s}/{:s}".format(tmp_path, name)
                    if not os.path.isdir(model_dir):
                        os.remove(model_dir)
                        continue
                    self.config["model_label"][model] = count
                    self.config["label_model"][count] = model
                    self.config["test_label_count"][str(count)] = 0
                    self.config["train_label_count"][str(count)] = 0
                    self.config["label_path"][str(count)] = "{:s}/{:s}".format(factory, name)
                    count += 1
            except Exception as e:
                print("=> [error] read_directory() e={:s}".format(e))
        self.config["label_count"] = count

        # push into self.config["train_list"]
        for (label, path) in self.config["label_path"].items():
            try:
                os.remove("{:s}/{:s}/.DS_Store".format(self.input_path, path))
            except Exception:
                pass

            tmp_list = os.listdir("{:s}/{:s}".format(self.input_path, path))
            self.config["train_list"][str(label)] = tmp_list
            self.config["train_label_count"][str(label)] = len(tmp_list)

        # push into self.config["test_list"]
        for (label, file_list) in self.config["train_list"].items():
            label = str(label)
            self.config["test_list"][label] = []

            if len(file_list) <= self.test_size:
                print("=> read_directory() {:s} not enough file".format(self.config["label_model"][int(label)]))
                continue

            # cross_instance verification: train instance x, and test instance y ...
            if self.cross_type == "instance":
                tmp_list = self.config["train_list"][label]
                self.config["train_list"][label] = []
                for file_name in tmp_list:
                    split_name = file_name.split("_")
                    instance_index = int(split_name[3]) if split_name[1] == "mju" else int(split_name[2])
                    if (instance_index % self.cross_rate) != 0:
                        self.config["test_label_count"][label] += 1
                        self.config["test_list"][label].append(file_name)
                        self.config["train_label_count"][label] -= 1
                    else:
                        self.config["train_list"][label].append(file_name)
            else:
                for i in range(self.test_size):
                    fid = random.randint(0, len(self.config["train_list"][label]) - 1)
                    self.config["test_list"][label].append(file_list[fid])
                    self.config["test_label_count"][label] += 1

                    # train && test dataset cross_validation
                    if random.uniform(0, 1) >= self.cross_rate:
                        del self.config["train_list"][label][fid]
                        self.config["train_label_count"][label] -= 1

        # write into file_list
        self.ModelDB.write(self.config)
        return self

    '''
    private function
    '''
    def read_and_decode(self, scope_name=None):
        scope_name = self.scope_name if scope_name == None else scope_name
        reader = tf.TFRecordReader()
        filename_queue = tf.train.string_input_producer(["{:s}/{:s}.tfrecords".format(self.output_path, scope_name)])
        (_, serialized_example) = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                "offset_x": tf.FixedLenFeature([], tf.int64),
                "offset_y": tf.FixedLenFeature([], tf.int64),
                "image_label": tf.FixedLenFeature([], tf.int64),
                "image_noise": tf.FixedLenFeature([], tf.float32),
                "image_data": tf.FixedLenFeature([], tf.string),
                "image_name": tf.FixedLenFeature([], tf.string)
            }
        )

        X = tf.reshape(tf.cast(tf.decode_raw(features["image_data"], tf.uint8), tf.float32), shape=self.model_conf["image_shape"])
        Y = tf.cast(features["image_label"], tf.int32)
        (name, image_noise) = (tf.cast(features["image_name"], tf.string), tf.cast(features["image_noise"], tf.float32))
        (offset_x, offset_y) = (tf.cast(features["offset_x"], tf.int32), tf.cast(features["offset_y"], tf.int32))
        return(X, Y, name, offset_x, offset_y, image_noise)

    '''
    [private function] crop image, we provide two method: 'rand croping' and 'slide windows croping' metods
    'rand croping': rand select [x, y], crop image by image[x: x+width, y: y+height]
    'slide croping': non-overloapping slide windows
    '''
    def crop(self, image, width, height, index):
        (x_start, x_stop) = (int(math.floor(width * self.model_conf["crop_dropout"])), int(math.floor(width * (1 - self.model_conf["crop_dropout"]))) - 64)
        (y_start, y_stop) = (int(math.floor(height * self.model_conf["crop_dropout"])), int(math.floor(height * (1 - self.model_conf["crop_dropout"]))) - 64)

        if self.generator_type == "rand":
            (_x, _y) = (random.randint(x_start, x_stop), random.randint(y_start, y_stop))
        else:
            max_x_size = (x_stop - x_start) / self.model_conf["image_shape"][0]
            _x = x_start + (index % max_x_size) * self.model_conf["image_shape"][0]
            _y = y_start + (index / max_x_size) * self.model_conf["image_shape"][1]
        crop_img = image[_x:_x + 64, _y:_y + 64, :]
        image_noise = np.var(crop_img)

        # only rand type care for noise
        if (self.generator_type == "rand") and (image_noise > self.max_image_noise):
            return ([], 0, 0, image_noise)
        else:
            return (crop_img, _x, _y, image_noise)

    def single_read(self, is_train=True):
        if not "file_path" in self.tmp_var:
            print("=> [error] please set variable: self.tmp_var['file_path']")
            return

        # we defined it, image_label == 0, this value is no use, becouse we don't know where is this photo come from.
        image_label = 0
        try:
            file_path = self.tmp_var["file_path"]
            image = misc.imread(file_path)
            (width, height) = (image.shape[0], image.shape[1])
            for index in range(self.batch_size):
                (crop_imgage_data, offset_x, offset_y, image_noise) = ([], 0, 0, 0)
                while offset_x == 0:
                    (crop_imgage_data, offset_x, offset_y, image_noise) = self.crop(image, width, height, index)
                yield (crop_imgage_data, image_label, file_path, offset_x, offset_y, image_noise)
        except Exception as e:
            print("=> [error] write() e={:s}".format(e))

    '''
    private function, iterator random read self.config["train_list"], yield return
    :return yield(imgage_data, image_label, image_name, offset_x, offset_y, image_noise)
    '''
    def iterator_read(self, is_train=True):
        self.read_directory()
        count = 1

        if (not is_train) and self.with_process:
            print("=> iterator_read() using compress, please run `code/util/compress.py` compress your images first.")

        while True:
            # choose an image from self.config["train_list"]
            file_list = self.config["train_list"] if is_train else self.config["test_list"]
            rand_label = str(random.randint(0, self.config["label_count"] - 1))

            # skip file_list[label] = null
            if len(file_list[rand_label]) == 0: continue

            rand_fid = random.randint(0, len(file_list[rand_label]) - 1)
            file_name = file_list[rand_label][rand_fid]
            model_path = self.config["label_path"][rand_label]
            full_name = "{:s}/{:s}".format(model_path, file_name)

            # using compress only for testset
            if (not is_train) and self.with_process:
                if not os.path.exists(full_name):
                    if self.process_type == "compress":
                        self.CompressUtil.compress_single("{:s}/{:s}".format(self.input_path, full_name))
                    # for other process method

                full_name = "{:s}/{:s}{:s}".format(model_path, self.process_prefix, file_name)

            try:
                if self.is_debug:
                    print("=> count={:d} model_id={:s} model={:s} file=/{:s}".format(count, rand_label, self.config["label_model"][int(rand_label)], full_name))
                image = misc.imread("{:s}/{:s}".format(self.input_path, full_name))
                (width, height) = (image.shape[0], image.shape[1])

                # crop self.batch_size image patchs
                for index in range(self.batch_size):
                    try_crop_times = 0
                    (crop_imgage_data, offset_x, offset_y, image_noise) = ([], 0, 0, 0)
                    while offset_x == 0:
                        try_crop_times += 1
                        if try_crop_times > self.model_conf["max_fail_times"]: exit(1)
                        (crop_imgage_data, offset_x, offset_y, image_noise) = self.crop(image, width, height, index)
                    yield(crop_imgage_data, int(rand_label), full_name, offset_x, offset_y, image_noise)
                count += 1
                del file_list[rand_label][rand_fid]
            except Exception as e:
                print(file_name, full_name)
                print(rand_label, rand_fid, len(file_list[rand_label]))
                print("=> count={:d} model_id={:s} model={:s} file=/{:s}".format(count, rand_label, self.config["label_model"][int(rand_label)], full_name))
                print("=> [error] iterator_read() e={:s}".format(e))
                exit(1)

            # if run all images: break loop
            tmp_count = 0
            for i in range(len(file_list)):
                if len(file_list[str(i)]) == 0:
                    tmp_count += 1
            if tmp_count == self.config["label_count"]: break
        print("=> iterator over")

    # ========================================public interface============================================== #

    '''
    public read interface for tfrecord file, don`t forget to iterator read by tf.Session()
    :param min_after_dequeue    int     queue cahce in memory, a control to shuffle_batch size
    :return [X, Y, name]    name is origin image name
    '''
    def read(self):
        capacity = self.min_after_dequeue * 64
        (_image, _label, _batch_name, _offset_x, _offset_y, _image_noise) = self.read_and_decode()
        (_X, _Y, _name, offset_x, offset_y) = tf.train.shuffle_batch([_image, _label, _batch_name, _offset_x, _offset_y], batch_size=self.batch_size, capacity=capacity, min_after_dequeue=self.min_after_dequeue)
        return (_X, _Y, _name, offset_x, offset_y)

    def iterator(self, session=None):
        # rebuild new tmp_data
        def rebuild():
            return {
                "offset_x": np.zeros([self.batch_size, 1]),
                "offset_y": np.zeros([self.batch_size, 1]),
                "image_label": np.zeros([self.batch_size]),
                "image_noise": np.zeros([self.batch_size, 1]),
                "image_name": [],
                "image_data": np.zeros([self.batch_size, self.model_conf["image_shape"][0], self.model_conf["image_shape"][1], self.model_conf["image_shape"][2]])
            }

        if type(session) == tf.Session:
            (G_X, G_Y, G_name, G_offset_x, G_offset_y) = self.read()
            tf.train.start_queue_runners(sess=session, coord=tf.train.Coordinator())
            for count in range(self.total_batch):
                (_X, _Y, _name, _x, _y) = session.run([G_X, G_Y, G_name, G_offset_x, G_offset_y])
                yield(count, (_X, _Y, _name, _x, _y))
        else:
            iterator_read = self.iterator_read
            if type(session) == type(" "):
                self.tmp_var["file_path"] = session
                iterator_read = self.single_read

            # if use_file_iterator, please make sure you have dataset in input_path
            count = 0
            tmp_data = rebuild()

            patch_count = -1
            for (image_data, image_label, image_name, offset_x, offset_y, image_noise) in iterator_read(is_train=False):
                patch_count += 1
                if patch_count == self.batch_size - 1:
                    # after a batch, yield return && reset all params
                    yield (count, (tmp_data["image_data"], np.array(tmp_data["image_label"], dtype=np.int32), tmp_data["image_name"], tmp_data["offset_x"], tmp_data["offset_y"]))
                    (count, patch_count, tmp_data) = (count + 1, 0, rebuild())

                tmp_data["offset_x"][patch_count] = offset_x
                tmp_data["offset_y"][patch_count] = offset_y
                tmp_data["image_label"][patch_count] = image_label
                tmp_data["image_data"][patch_count]= image_data #np.reshape(image_data, [config["SHAPE"][0] * config["SHAPE"][1] * config["SHAPE"][2]])
                tmp_data["image_name"].append(image_name)
                tmp_data["image_noise"][patch_count] = image_noise

    '''
    [public funtion] write file into tfrecord, interface from self.iterator_read()
    :param [image_data, image_label, image_name, offset_x, offset_y, image_noise]
    '''
    def write(self, scope_name=None):
        scope_name = self.scope_name if scope_name == None else scope_name
        self.writer = tf.python_io.TFRecordWriter("{:s}/{:s}.tfrecords".format(self.output_path, scope_name))
        for (image_data, image_label, image_name, offset_x, offset_y, image_noise) in self.iterator_read():
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "offset_x": self._int64_feature(offset_x),
                        "offset_y": self._int64_feature(offset_y),
                        "image_label": self._int64_feature(image_label),
                        "image_noise": self._float32_feature(image_noise),
                        "image_data": self._bytes_feature(image_data.tostring()),
                        "image_name": self._bytes_feature(image_name)
                    }
                )
            )
            self.writer.write(example.SerializeToString())
if __name__ == "__main__":
    import json
    with open("conf.json", "r") as f:
        conf = json.load(f)
        G = Generator(conf=conf, is_debug=True)
        G.write()
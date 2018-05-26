#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#                  ___====-_  _-====___
#            _--^^^#####//      \\#####^^^--_
#         _-^##########// (    ) \\##########^-_
#        -############//  |\^^/|  \\############-
#      _/############//   (@::@)   \\############\_
#     /#############((     \\//     ))#############\
#    -###############\\    (oo)    //###############-
#   -#################\\  / VV \  //#################-
#  -###################\\/      \//###################-
# _#/|##########/\######(   /\   )######/\##########|\#_
# |/ |#/\#/\#/\/  \#/\##\  |  |  /##/\#/  \/\#/\#/\#| \|
# `  |/  V  V  `   V  \#\| |  | |/#/  V   '  V  V  \|  '
#    `   `  `      `   / | |  | | \   '      '  '   '
#                     (  | |  | |  )
#                    __\ | |  | | /__
#                   (vvv(VVV)(VVV)vvv)
#
#               神兽保佑            永不修改
#                        `=---='
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
__author__ = 'grasses'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__   = 'Copyright © 2017/10/31, grasses'

import json
from network import Network
from generator import Generator
with open("conf.json", "r") as f:
    conf = json.load(f)

G = Generator(conf=conf)
(G_X, G_Y, G_name, G_offset_x, G_offset_y) = G.read()

def main():
    N = Network(conf=conf)
    N.define_inputs()
    N.add_conv(name='conv1', patch_size=8, in_depth=3, out_depth=16, stride=[1, 2, 2, 1], activation='relu', pooling=False)
    N.add_conv(name='conv2', patch_size=8, in_depth=16, out_depth=32, stride=[1, 1, 1, 1], activation='relu', pooling=False)
    N.add_conv(name='conv3', patch_size=6, in_depth=32, out_depth=48, stride=[1, 1, 1, 1], activation='relu', pooling=False)
    N.add_conv(name='conv4', patch_size=6, in_depth=48, out_depth=64, stride=[1, 2, 2, 1], activation='relu', pooling=False)
    N.add_conv(name='conv5', patch_size=3, in_depth=64, out_depth=128, stride=[1, 1, 1, 1], activation='relu', pooling=False)
    N.add_conv(name='conv6', patch_size=3, in_depth=128, out_depth=256, stride=[1, 1, 1, 1], activation='relu', pooling=True)
    N.add_conv(name='conv7', patch_size=3, in_depth=256, out_depth=512, stride=[1, 2, 2, 1], activation='relu', pooling=False)
    N.add_conv(name='conv8', patch_size=3, in_depth=512, out_depth=1024, stride=[1, 1, 1, 1], activation='relu', pooling=False)
    N.add_conv(name='conv9', patch_size=1, in_depth=1024, out_depth=512, stride=[1, 1, 1, 1], activation='relu', pooling=True)
    N.add_conv(name='conv10', patch_size=1, in_depth=512, out_depth=512, stride=[1, 2, 2, 1], activation='relu', pooling=False)
    N.add_conv(name='conv11', patch_size=1, in_depth=512, out_depth=256, stride=[1, 1, 1, 1], activation='relu', pooling=False)
    N.add_conv(name='conv12', patch_size=1, in_depth=256, out_depth=128, stride=[1, 2, 2, 1], activation='relu', pooling=False)
    N.add_conv(name='conv13', patch_size=1, in_depth=128, out_depth=64, stride=[1, 2, 2, 1], activation='relu', pooling=False)

    N.add_fc(name='fc1', in_num_nodes=64, out_num_nodes=32, activation='relu')
    N.add_fc(name='fc2', in_num_nodes=32, out_num_nodes=64, activation='relu')
    N.add_fc(name='fc3', in_num_nodes=64, out_num_nodes=conf["net"]["label_count"], activation=None)
    N.train(G_X, G_Y, G_name)

if __name__ == "__main__":
    main()

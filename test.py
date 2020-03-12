
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from ops import *
import cv2
import os
from vgg import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from natsort import natsorted
import imageio

path = 'data'

np.random.seed(0)

global params
params = {'path' : path,
          'batch_size' : 8,
          'output_size': 256,
          'gf_dim': 32,
          'df_dim': 32,
          'model_path' : './model',
          'L1_lambda': 100,
          'lr': 0.0001,
          'beta_1': 0.5,
          'epochs': 50,
          'test_output' : './tests/output',
          'Stage_epochs' : [10000,20000]}

if not os.path.isdir(params['test_output']):
    os.mkdir(params['test_output'])
    
def get_file_paths(path):
    img_paths = [os.path.join(root, file)  for root, dirs, files in os.walk(path) for file in files if '_gt' not in file]
    gt_path = [os.path.join(os.path.dirname(file), os.path.basename(file).split('.')[0] + '_gt.png') for file in img_paths]
    return np.array(img_paths[:int(len(img_paths)*.9)]), np.array(gt_path[:int(len(gt_path)*.9)]), np.array(img_paths[int(len(img_paths)*.9):])        , np.array(gt_path[int(len(gt_path)*.9):])


def load_data_CONTENT(path):
    im = ~cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (256, 256))
    return np.expand_dims(im, -1)/127.5 - 1.

def load_data(path):
    im = cv2.resize(cv2.imread(path, 0), (256, 256))
    return  np.expand_dims(im, -1)/127.5 - 1.

# Functions to load and save weights
def load_weights(saver, model_dir):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(model_dir, ckpt_name))
        print("MODEL LOADED SUCCESSFULLY")
    else:
        print("LOADING MODEL FAILED")

def save(saver, checkpoint_dir, step):
    dir = os.path.join(checkpoint_dir, "model")
    saver.save(sess, dir, step)


global g_bn_d1, g_bn_d2, g_bn_d3, g_bn_d4, g_bn_d5, g_bn_d6, g_bn_d7

d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')

g_bn_e2 = batch_norm(name='g_bn_e2')
g_bn_e3 = batch_norm(name='g_bn_e3')
g_bn_e4 = batch_norm(name='g_bn_e4')
g_bn_e5 = batch_norm(name='g_bn_e5')
g_bn_e6 = batch_norm(name='g_bn_e6')
g_bn_e7 = batch_norm(name='g_bn_e7')
g_bn_e8 = batch_norm(name='g_bn_e8')

g_bn_d1 = batch_norm(name='g_bn_d1')
g_bn_d2 = batch_norm(name='g_bn_d2')
g_bn_d3 = batch_norm(name='g_bn_d3')
g_bn_d4 = batch_norm(name='g_bn_d4')
g_bn_d5 = batch_norm(name='g_bn_d5')
g_bn_d6 = batch_norm(name='g_bn_d6')
g_bn_d7 = batch_norm(name='g_bn_d7')


global g_bn_d1_, g_bn_d2_, g_bn_d3_, g_bn_d4_, g_bn_d5_, g_bn_d6_, g_bn_d7_


global d_bn1_, d_bn2_, d_bn3_
d_bn1_ = batch_norm(name='d_bn1_')
d_bn2_ = batch_norm(name='d_bn2_')
d_bn3_ = batch_norm(name='d_bn3_')

g_bn_e2_ = batch_norm(name='g_bn_e2_')
g_bn_e3_ = batch_norm(name='g_bn_e3_')
g_bn_e4_ = batch_norm(name='g_bn_e4_')
g_bn_e5_ = batch_norm(name='g_bn_e5_')
g_bn_e6_ = batch_norm(name='g_bn_e6_')
g_bn_e7_ = batch_norm(name='g_bn_e7_')
g_bn_e8_ = batch_norm(name='g_bn_e8_')

g_bn_d1_ = batch_norm(name='g_bn_d1_')
g_bn_d2_ = batch_norm(name='g_bn_d2_')
g_bn_d3_ = batch_norm(name='g_bn_d3_')
g_bn_d4_ = batch_norm(name='g_bn_d4_')
g_bn_d5_ = batch_norm(name='g_bn_d5_')
g_bn_d6_ = batch_norm(name='g_bn_d6_')
g_bn_d7_ = batch_norm(name='g_bn_d7_')


def Noise_transfer_network(content, style, y=None):
    s = params['output_size']
    output_c_dim = 1
    s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
    gf_dim = params['gf_dim']
    
    
    with tf.variable_scope("generator1") as globscope:
    
        with tf.variable_scope("content_encoder") as scope:

            # image is (256 x 256 x input_c_dim)
            c_e1 = conv2d(content, gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x gf_dim)
            c_e2 = g_bn_e2(conv2d(lrelu(c_e1), gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x gf_dim*2)
            c_e3 = g_bn_e3(conv2d(lrelu(c_e2), gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x gf_dim*4)
            c_e4 = g_bn_e4(conv2d(lrelu(c_e3), gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x gf_dim*8)
            c_e5 = g_bn_e5(conv2d(lrelu(c_e4), gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x gf_dim*8)
            c_e6 = g_bn_e6(conv2d(lrelu(c_e5), gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x gf_dim*8)
            c_e7 = g_bn_e7(conv2d(lrelu(c_e6), gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x gf_dim*8)
            c_e8 = g_bn_e8(conv2d(lrelu(c_e7), gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x gf_dim*8)

        with tf.variable_scope("style_encoder") as scope:

            # image is (256 x 256 x input_c_dim)
            s_e1 = conv2d(style, gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x gf_dim)
            s_e2 = g_bn_e2(conv2d(lrelu(s_e1), gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x gf_dim*2)
            s_e3 = g_bn_e3(conv2d(lrelu(s_e2), gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x gf_dim*4)
            s_e4 = g_bn_e4(conv2d(lrelu(s_e3), gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x gf_dim*8)
            s_e5 = g_bn_e5(conv2d(lrelu(s_e4), gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x gf_dim*8)
            s_e6 = g_bn_e6(conv2d(lrelu(s_e5), gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x gf_dim*8)
            s_e7 = g_bn_e7(conv2d(lrelu(s_e6), gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x gf_dim*8)
            s_e8 = g_bn_e8(conv2d(lrelu(s_e7), gf_dim*8, name='g_e8_conv'))

        m_e8 = tf.concat([c_e8, s_e8],-1)

        with tf.variable_scope("decoder") as scope:

            batch_size = params['batch_size']
            d1, d1_w, d1_b = deconv2d(tf.nn.relu(m_e8),
                [batch_size, s128, s128, gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(g_bn_d1(d1), 0.5)
            d1 = tf.concat([d1, c_e7], 3)
            # d1 is (2 x 2 x gf_dim*8*2)

            d2, d2_w, d2_b = deconv2d(tf.nn.relu(d1),
                [batch_size, s64, s64, gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(g_bn_d2(d2), 0.5)
            d2 = tf.concat([d2, c_e6], 3)
            # d2 is (4 x 4 x gf_dim*8*2)

            d3, d3_w, d3_b = deconv2d(tf.nn.relu(d2),
                [batch_size, s32, s32, gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(g_bn_d3(d3), 0.5)
            d3 = tf.concat([d3, c_e5], 3)
            # d3 is (8 x 8 x gf_dim*8*2)

            d4, d4_w, d4_b = deconv2d(tf.nn.relu(d3),
                [batch_size, s16, s16, gf_dim*8], name='g_d4', with_w=True)
            d4 = g_bn_d4(d4)
            d4 = tf.concat([d4, c_e4], 3)
            # d4 is (16 x 16 x gf_dim*8*2)

            d5, d5_w, d5_b = deconv2d(tf.nn.relu(d4),
                [batch_size, s8, s8, gf_dim*4], name='g_d5', with_w=True)
            d5 = g_bn_d5(d5)
            d5 = tf.concat([d5, c_e3], 3)
            # d5 is (32 x 32 x gf_dim*4*2)

            d6, d6_w, sd6_b = deconv2d(tf.nn.relu(d5),
                [batch_size, s4, s4, gf_dim*2], name='g_d6', with_w=True)
            d6 = g_bn_d6(d6)
            d6 = tf.concat([d6, c_e2], 3)
            # d6 is (64 x 64 x gf_dim*2*2)

            d7, d7_w, d7_b = deconv2d(tf.nn.relu(d6),
                [batch_size, s2, s2, gf_dim], name='g_d7', with_w=True)
            d7 = g_bn_d7(d7)
            d7 = tf.concat([d7, c_e1], 3)
            # d7 is (128 x 128 x gf_dim*1*2)

            d8, d8_w, d8_b = deconv2d(tf.nn.relu(d7),
                [batch_size, s, s, output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(d8)



def Noise_remover_network(image, y=None):
    
    s = params['output_size']
    output_c_dim = 1
    s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
    gf_dim = params['gf_dim']
        
    with tf.variable_scope("generator2", reuse = tf.AUTO_REUSE) as scope:

        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(image, gf_dim, name='g_e1_conv')
        # e1 is (128 x 128 x gf_dim)
        e2 = g_bn_e2_(conv2d(lrelu(e1), gf_dim*2, name='g_e2_conv'))
        # e2 is (64 x 64 x gf_dim*2)
        e3 = g_bn_e3_(conv2d(lrelu(e2), gf_dim*4, name='g_e3_conv'))
        # e3 is (32 x 32 x gf_dim*4)
        e4 = g_bn_e4_(conv2d(lrelu(e3), gf_dim*8, name='g_e4_conv'))
        # e4 is (16 x 16 x gf_dim*8)
        e5 = g_bn_e5_(conv2d(lrelu(e4), gf_dim*8, name='g_e5_conv'))
        # e5 is (8 x 8 x gf_dim*8)
        e6 = g_bn_e6_(conv2d(lrelu(e5), gf_dim*8, name='g_e6_conv'))
        # e6 is (4 x 4 x gf_dim*8)
        e7 = g_bn_e7_(conv2d(lrelu(e6), gf_dim*8, name='g_e7_conv'))
        # e7 is (2 x 2 x gf_dim*8)
        e8 = g_bn_e8_(conv2d(lrelu(e7), gf_dim*8, name='g_e8_conv'))
        # e8 is (1 x 1 x gf_dim*8)
        
        batch_size = params['batch_size']
        d1, d1_w, d1_b = deconv2d(tf.nn.relu(e8),
            [batch_size, s128, s128, gf_dim*8], name='g_d1', with_w=True)
        d1 = tf.nn.dropout(g_bn_d1_(d1), 0.5)
        d1 = tf.concat([d1, e7], 3)
        # d1 is (2 x 2 x gf_dim*8*2)

        d2, d2_w, d2_b = deconv2d(tf.nn.relu(d1),
            [batch_size, s64, s64, gf_dim*8], name='g_d2', with_w=True)
        d2 = tf.nn.dropout(g_bn_d2_(d2), 0.5)
        d2 = tf.concat([d2, e6], 3)
        # d2 is (4 x 4 x gf_dim*8*2)

        d3, d3_w, d3_b = deconv2d(tf.nn.relu(d2),
            [batch_size, s32, s32, gf_dim*8], name='g_d3', with_w=True)
        d3 = tf.nn.dropout(g_bn_d3_(d3), 0.5)
        d3 = tf.concat([d3, e5], 3)
        # d3 is (8 x 8 x gf_dim*8*2)

        d4, d4_w, d4_b = deconv2d(tf.nn.relu(d3),
            [batch_size, s16, s16, gf_dim*8], name='g_d4', with_w=True)
        d4 = g_bn_d4_(d4)
        d4 = tf.concat([d4, e4], 3)
        # d4 is (16 x 16 x gf_dim*8*2)

        d5, d5_w, d5_b = deconv2d(tf.nn.relu(d4),
            [batch_size, s8, s8, gf_dim*4], name='g_d5', with_w=True)
        d5 = g_bn_d5_(d5)
        d5 = tf.concat([d5, e3], 3)
        # d5 is (32 x 32 x gf_dim*4*2)

        d6, d6_w, sd6_b = deconv2d(tf.nn.relu(d5),
            [batch_size, s4, s4, gf_dim*2], name='g_d6', with_w=True)
        d6 = g_bn_d6_(d6)
        d6 = tf.concat([d6, e2], 3)
        # d6 is (64 x 64 x gf_dim*2*2)

        d7, d7_w, d7_b = deconv2d(tf.nn.relu(d6),
            [batch_size, s2, s2, gf_dim], name='g_d7', with_w=True)
        d7 = g_bn_d7_(d7)
        d7 = tf.concat([d7, e1], 3)
        # d7 is (128 x 128 x gf_dim*1*2)

        d8, d8_w, d8_b = deconv2d(tf.nn.relu(d7),
            [batch_size, s, s, output_c_dim], name='g_d8', with_w=True)
        # d8 is (256 x 256 x output_c_dim)

        #return tf.nn.tanh(d8[:,:,:,:3]), tf.nn.tanh(d8[:,:,:,3:4])  #(w/o text , bin text)
        return tf.nn.tanh(d8)

from tensorflow.python.framework import ops
ops.reset_default_graph()

global sess

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session()
graph = tf.get_default_graph()

content = tf.placeholder(dtype=tf.float32, shape=[params['batch_size'],256,256,1], name = 'content')
style = tf.placeholder(dtype=tf.float32, shape=[params['batch_size'],256,256,1], name = 'style')
Real_input = tf.placeholder(dtype=tf.float32, shape=[params['batch_size'],256,256,1], name = 'Real_input')

output = Noise_transfer_network(content, style)

Cleaned = Noise_remover_network(output)

# For testing
Real_cleaned = Noise_remover_network(Real_input)


init_op = tf.global_variables_initializer()
sess.run(init_op)

saver = tf.train.Saver()
load_weights(saver, params['model_path'])


# def load_data_CONTENT(path):
    # return np.expand_dims(im, -1)/127.5 - 1.
    # im = ~cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (256, 256))

import matplotlib.pyplot as plt
from pathlib import Path

cc = 1
all_inputs = list(Path("tests/input").glob("*.jp*"))
all_inputs = [str(l) for l in all_inputs]

from math import ceil

for l in all_inputs:
    im = cv2.imread(l, cv2.IMREAD_GRAYSCALE)
    new_h = ceil(im.shape[0] / 256.0) * 256
    new_w = ceil(im.shape[1] / 256.0) * 256
    new_im = np.ones((new_h,new_w), dtype=np.uint8) * 255
    new_im[:im.shape[0], :im.shape[1]] = im
    # print(im.shape)
    # print(new_im.shape)
    # plt.imshow(new_im)
    # plt.show()
    crops = list()
    for i in range(0,new_h,256):
        for j in range(0,new_w,256):
            crop = new_im[i:i+256,j:j+256]
            crop = np.expand_dims(crop, -1)/127.5 - 1.
            crops.append((crop, (i, j)))

    if len(crops) % params['batch_size'] != 0:
        need = params['batch_size'] - len(crops) % params['batch_size']
        while need > 0:
            blank = np.ones( (256,256,1),dtype=np.uint8) * 256/127.5 - 1
            crops.append((blank, None))
            need -= 1

    batches_cnt = (len(crops)+params['batch_size']-1) // params['batch_size']

    output = np.zeros(new_im.shape, dtype=np.float32)
    print("sdfsdf", output.shape)

    for k in range(batches_cnt):
        batchx_data_raw = crops[k * params['batch_size'] : (k + 1) * params['batch_size']]
        batchx_data = [x[0] for x in batchx_data_raw]

        feed_dict = {Real_input : batchx_data}
        
        Real_cleaned_ = sess.run(Real_cleaned, feed_dict)

        # all_imgs = np.concatenate((batchx_data, Real_cleaned_),2)
        all_imgs = Real_cleaned_

        for no_i in range(params['batch_size']):    
            # imageio.imwrite(os.path.join(params['test_output'],str(cc) + ".jpg"), all_imgs[no_i,:,:,:])
            # cc = cc + 1
            if batchx_data_raw[no_i][1] is None:
                continue

            i, j = batchx_data_raw[no_i][1]
            img =  np.reshape(all_imgs[no_i,:,:,:], (256,256))
            print(output.shape, i+256,j+256)
            output[i:i+256,j:j+256] = np.reshape(all_imgs[no_i,:,:,:], (256,256))
    cc += 1
    output = np.concatenate((im/127.5 - 1, output[:im.shape[0], :im.shape[1]]), 1)
    imageio.imwrite(os.path.join(params['test_output'],str(cc) + ".jpg"),output)
import tensorflow as tf
from tensorflow.examples.tutorials import mnist
import numpy as np
import cv2

z = tf.placeholder(tf.float32,shape=[None,100])
x = tf.placeholder(tf.float32,shape=[None,784])
batch_size=128
mnist = mnist.input_data.read_data_sets('./mnist')

def generate(z):
    net = tf.layers.dense(z,128,activation=tf.nn.leaky_relu,name='linear_1')
    net = tf.layers.dense(net,784,name='linear_2')  #28*28
    net_prob = tf.nn.sigmoid(net)
    return net_prob

def discriminate(x):
    x = tf.layers.flatten(x)
    net = tf.layers.dense(x,128,activation=tf.nn.leaky_relu,name='linear_1')
    net = tf.layers.dense(net,1,name='linear_2')
    net_prob = tf.nn.sigmoid(net)
    return net, net_prob

def sample_z(dim):
    return np.random.uniform(-1,1,[dim,100])

with tf.variable_scope('g'):
    gz_prob = generate(z)

with tf.variable_scope('d',reuse=tf.AUTO_REUSE):
    dx, dx_prob = discriminate(x)
    d_gz, d_gz_prob = discriminate(gz_prob)

g_loss = -tf.reduce_mean(tf.log(d_gz_prob))
d_loss = -tf.reduce_mean(tf.log(dx_prob)+tf.log(1-d_gz_prob))

var_list_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='g')
var_list_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='d')

d_slover = tf.train.AdamOptimizer(0.0001).minimize(d_loss,var_list=var_list_d)
g_slover = tf.train.AdamOptimizer().minimize(g_loss,var_list=var_list_g)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000000):
        x_, _ = mnist.train.next_batch(batch_size)   # x_ in [0-1]
        x_ = x_.reshape([batch_size,-1])
        _, loss_d = sess.run([d_slover,d_loss],feed_dict={z:sample_z(batch_size),
                                              x:x_})
        if i %1 == 0:
            _, loss_g = sess.run([g_slover,g_loss],feed_dict={z:sample_z(batch_size)})

        if i %100 == 0:
            g_z , d_g_z_prob = sess.run([gz_prob,d_gz_prob],feed_dict={z:sample_z(batch_size)})
            img = np.reshape(g_z,[batch_size,28,28])
            img = np.uint8(img*255.0)
            cv2.imshow('g0',img[0])
            cv2.imshow('g1',img[1])
            cv2.waitKey(100)
            print('%d iterations g_loss is %f'% (i,loss_g))
            print('%d iterations d_loss is %f'% (i, loss_d))
            print('output is %f'%(d_g_z_prob.sum()/batch_size))
            print('\n')


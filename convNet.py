# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:17:25 2019

@author: Lucas
"""

# Basic implementation of a CNN using tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Importing the nmist dataset
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('/tmp/data', one_hot = True)

# Defining the convolutional network structure
inputSize = 1
filtersConv1 = 32
filtersConv2 = 64

# Defining the fully conected network structure
inputLayerSize = 3136 # output filter dimension(7x7) * 64 filters
hiddenLayerSize = 512
outputLayerSize = 10

# Initizalizing weights (filters for conv layers) and biases
wc1 = tf.Variable(tf.random_normal([3,3,inputSize,filtersConv1]))
bc1 = tf.Variable(tf.random_normal([filtersConv1]))
wc2 = tf.Variable(tf.random_normal([3,3,filtersConv1,filtersConv2]))
bc2 = tf.Variable(tf.random_normal([filtersConv2]))

wl1 = tf.Variable(tf.random_normal([inputLayerSize, hiddenLayerSize]))
bl1 = tf.Variable(tf.random_normal([hiddenLayerSize]))
wl2 = tf.Variable(tf.random_normal([hiddenLayerSize, outputLayerSize]))
bl2 = tf.Variable(tf.random_normal([outputLayerSize]))

# Defining convolution and pooling function
def conv2D(X, w):
  return tf.nn.conv2d(X, w, [1,1,1,1], padding='SAME')

def maxpool2D(X):
  return tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Defining a feed forward function to return an output
def conv_neural_network(X, mode):
  X = tf.reshape(X, shape=[-1, 28, 28, 1])
  conv1 = conv2D(X,wc1)
  conv1 = tf.nn.bias_add(conv1, bc1)
  conv1act = tf.nn.relu(conv1)
  convl1 = maxpool2D(conv1act)
  
  conv2 = conv2D(convl1, wc2)
  conv1 = tf.nn.bias_add(conv2, bc2)
  conv2act = tf.nn.relu(conv2)
  convl2 = maxpool2D(conv2act)
  
  flattened = tf.reshape(convl2,[-1, inputLayerSize])
  
  z2 = tf.add(tf.matmul(flattened, wl1), bl1)
  a2 = tf.nn.relu(z2)
  a2 = tf.nn.dropout(a2, rate=0.1)
  
  y_hat = tf.add(tf.matmul(a2, wl2), bl2)
  
  if(mode == 't'):
    return y_hat
  
  y_hat_softmax = tf.nn.softmax(y_hat)
  
  return y_hat_softmax

# Initializing placeholders to be replaced with X and y values on training
xx = tf.placeholder("float")
yy = tf.placeholder("float")

# Output and cost functions
y_hat = conv_neural_network(xx, mode = 't')
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat,labels=yy))

# Using Adam optimizer for backprop with a learning rate of 0.001
train = tf.train.AdamOptimizer(0.001).minimize(cost)

# Empty list to be filled with cost values
train_cost = []
val_cost = []
acc_array = []
n_epochs = 30
batch_size = 50
iters = data.train.num_examples/batch_size

# Setup the initialisation operator
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

# Opening new tensorflow session to train the network
with tf.Session() as sess:
  sess.run(init_op)
  Xval = data.validation.images
  yval = data.validation.labels
  
  for i in range(n_epochs):
    c_train = 0
    for j in range(int(iters)):
      Xtrain, ytrain = data.train.next_batch(batch_size)
      c_train_batch, _ = sess.run([cost, train], feed_dict={xx:Xtrain, yy:ytrain})
      c_train += (c_train_batch / (iters))
    
    train_cost.append(c_train)
    val_cost.append(sess.run(cost, feed_dict={xx:Xval, yy:yval}))
    print('Epoch:',i,'Cost:', train_cost[i], 'Val Cost:', val_cost[i])
    
    
    correct = tf.equal(tf.math.argmax(y_hat, 1), tf.math.argmax(yy, 1))
    accuracy = tf.math.reduce_mean(tf.cast(correct, 'float'))
    acc_array.append(accuracy.eval({xx:data.test.images, yy:data.test.labels}))
    print('Accuracy:',acc_array[i])
  
  #Save model (optional)  
  #save_path = saver.save(sess, "model.ckpt")
  #print ("Model saved in file: ", save_path)

# Visualizing training and validation cost curves
xaxis = np.linspace(1, n_epochs, n_epochs-1)
plt.plot(xaxis, train_cost[1:], label='Train cost')
plt.plot(xaxis, val_cost[1:], label='Test Cost')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.legend()
plt.grid()
plt.show()

# Visualize accuracy
xaxis = np.linspace(0, n_epochs, n_epochs)
plt.plot(xaxis, acc_array)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.show()

#Visualize example from dataset
#Xdata = data.test.images
#photo = Xdata[741].reshape(28,28)
#plt.imshow(photo, cmap='Greys')


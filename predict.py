# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:48:54 2019

@author: Lucas
"""
# Importing libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Setup the initialisation operator
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

# Import and formating custom photo
from skimage import color
from skimage import io
custom_photo = color.rgb2gray(io.imread('samples/three.jpg'))
custom_photo = np.array(custom_photo)
custom_photo = np.float32(custom_photo)
custom_photo = 1 - custom_photo
custom_photo[custom_photo < 0.05] = 0 
plt.imshow(custom_photo, cmap='Greys')

# Prediction code laoding saved model
with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "model.ckpt")
        prediction=sess.run(conv_neural_network(custom_photo, mode='p'))
print('The number is:', np.argmax(prediction))
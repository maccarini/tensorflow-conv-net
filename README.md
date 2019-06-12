# Tensorflow Convolutional Neural Network
A basic implementation of an artificial neural network on python using numpy, matplotlib and tensorflow.

## The dataset
The dataset used for this project is the mnist dataset. It consists of 60000, 28 x 28 samples of handwritten digits which are used to train the network. 
A sample from the dataset can be visualized as follows
```python
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('/tmp/data', one_hot = True)

Xdata = data.test.images
sample = Xdata[741].reshape(28,28)
plt.imshow(sample, cmap='Greys')
```
![Sample-data](https://github.com/maccarini/tensorflow-conv-net/blob/master/samples/sample_data.png "Sample digit")

Further information about this dataset is available on [Kaggle](https://www.kaggle.com/c/digit-recognizer/data). 

## The problem
The main goal of this neural network is to predict handwitten digits with the highest possible accuracy, for this we will use a convolutional neural network consisting of 2 convolutional layers (with reLU as activation function), and one hidden layer for the fully conected layer and a 0.1 dropout rate for that layer.

## Results
After training for 30 epochs we observe that the training and validation costs decrease as expected obtain an accuracy of aproximately 98.5 which does not increase majorly if we increase the epoch number.

![train-test-curve](https://github.com/maccarini/tensorflow-conv-net/blob/master/train_test_cost.png "Train-test curve")
![accuracy-curve](https://github.com/maccarini/tensorflow-conv-net/blob/master/accuracy.png "Accuracy")

We can check that it also predicts digits that are not contained on the dataset itself, for this, a sample folder is included that contains digits made on MS Paint.
```python
from skimage import color
from skimage import io
custom_photo = color.rgb2gray(io.imread('samples/three.jpg'))
custom_photo = np.array(custom_photo)
custom_photo = np.float32(custom_photo)
custom_photo = 1 - custom_photo
custom_photo[custom_photo < 0.05] = 0 
```
```python
with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "model.ckpt")
        prediction=sess.run(conv_neural_network(custom_photo, mode='p'))
print('The number is:', np.argmax(prediction))
```
And it outputs
```
The number is: 3
```

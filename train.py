from tensorflow.examples.tutorials.mnist import input_data
from generator import * 
import tensorflow as tf

# read data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

## Tensor definition
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

generator = Generator()
y_conv = generator(input_tensor=x, teacher_tensor=y_)

# loss definition
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# additional tensor definition for training estimation
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train definition
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# training loop
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(20000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1]})
    print("step %d, training accuracy %g"%(i, train_accuracy))

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels}))

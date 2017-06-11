import tensorflow as tf

class Generator():
  def __init__(self):
    pass

  def __call__(self, input_tensor, teacher_tensor):
    x = input_tensor
    y_ = teacher_tensor
    
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    
    y = tf.matmul(x,W) + b
    
    W_conv1 = self.weight_variable([5, 5, 1, 32])
    b_conv1 = self.bias_variable([32])
    
    x_image = tf.reshape(x, [-1,28,28,1])
    
    h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = self.max_pool_2x2(h_conv1)
    
    W_conv2 = self.weight_variable([5, 5, 32, 64])
    b_conv2 = self.bias_variable([64])
    
    h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = self.max_pool_2x2(h_conv2)
    
    W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
    b_fc1 = self.bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    #keep_prob = tf.placeholder(tf.float32)
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = self.weight_variable([1024, 10])
    b_fc2 = self.bias_variable([10])
    
    #y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv
    
  def conv2d(self, x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  
  def max_pool_2x2(self, x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
  
  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
  
  def bias_variable(self, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
  

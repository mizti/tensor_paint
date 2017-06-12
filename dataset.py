import tensorflow as tf
import numpy as np

class PaintData():
  def __init__(self):
    pass

  def get_dataset(self):
    # creating Tensor data
    #filelist = ["data/1.jpg", "data/2.jpg", "data/3.jpg"]
    #filequeue = tf.train.string_input_producer(filelist)
    
    #data = tf.random_uniform([i], -1, i)
    #label = tf.cast(tf.reduce_prod(data)>=0, tf.float32)
    data = np.asarray([3,4,5])
    label = np.asarray([1,2,3])

    # batch data definition 
    X_batch, Y_batch = tf.train.batch([data, label], batch_size=10)
    with tf.Session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      XData, YData = sess.run([X_batch, Y_batch])
      return XData, YData
#    with tf.Session() as sess:
#      coord = tf.train.Coordinator()
#      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#    
#      try:
#        while not coord.should_stop():
#          XData, YData = sess.run([X_batch, Y_batch])
#          print(XData.shape)
#    
#      finally:
#        coord.request_stop()
#        coord.join(threads)
#
d = PaintData()
input, teacher = d.get_dataset()
print(input)
print(teacher)

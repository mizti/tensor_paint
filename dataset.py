import tensorflow as tf
import numpy as np

class PaintData():
  def __init__(self):
    pass
  # this may be a help: https://gist.github.com/leVirve/80428277f2d14515870fa18899fed17a
  def get_dataset(self):
    #filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once('*'), num_epochs=1, shuffle=True)
    #filenames = tf.train.match_filenames_once('./data/*')
    filenames = tf.train.match_filenames_once('data/*.jpg')

    #with tf.Session() as sess:
    #  #tf.global_variables_initializer().run()
    #  tf.local_variables_initializer().run() # why not global?
    #  print(sess.run(filenames))
    #  print('======') 
    #exit()
    filename_queue = tf.train.string_input_producer(filenames)

    reader = tf.WholeFileReader()
    #_, value = reader.read(filename_queue)
    _, value = reader.read(filename_queue)
    
    image = tf.image.decode_jpeg(value, channels=3)
    image = tf.image.resize_images(image, [100, 100])
    
    with tf.Session() as sess:
    
      #tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
    
      image_tensor = sess.run([image])
      print(image_tensor)
    
      coord.request_stop()
      coord.join(threads)
      return image_tensor, "hoge"

  def get_dataset2(self):
    # creating Tensor data
    filelist = ["data/1.jpg", "data/2.jpg", "data/3.jpg"]
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

import tensorflow as tf
import numpy as np

class PaintData():
  def __init__(self):
    self.filenames = tf.train.match_filenames_once('data/*.jpg')
    self.filename_queue = tf.train.string_input_producer(self.filenames)
    self.reader = tf.WholeFileReader()
    _, self.value = self.reader.read(self.filename_queue)
    self.image = tf.image.decode_jpeg(self.value, channels=3)
    self.image = tf.image.resize_images(self.image, [100, 100])

  # this may be a help: https://gist.github.com/leVirve/80428277f2d14515870fa18899fed17a
  def get_dataset(self):
    #filenames = tf.train.match_filenames_once('data/*.jpg')
    #filename_queue = tf.train.string_input_producer(filenames)
    #reader = tf.WholeFileReader()
    #_, value = reader.read(filename_queue)
    #image = tf.image.decode_jpeg(value, channels=3)
    #image = tf.image.resize_images(image, [100, 100])
    
    with tf.Session() as sess:
      #tf.global_variables_initializer().run() # why not global?
      tf.local_variables_initializer().run()
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      image_tensor = sess.run([self.image])
      coord.request_stop()
      coord.join(threads)
      return image_tensor, "hoge"

d = PaintData()
input, teacher = d.get_dataset()
input = reduce(lambda x,y: x+y, input) / len(input)
input = reduce(lambda x,y: x+y, input) / len(input)
input = reduce(lambda x,y: x+y, input) / len(input)
input = reduce(lambda x,y: x+y, input) / len(input)
print(input)

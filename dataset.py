import tensorflow as tf

def get_dataset():
    filename_queue = tf.train.string_input_producer(["filelist.csv"])
    
    reader = tf.TextLineReader()
    
    key = reader.read(filename_queue)
    
    #data = tf.random_uniform([2], -1, 1)
    #label = tf.cast(tf.reduce_prod(data)>=0, tf.float32)
    
    #X_batch, Y_batch = tf.train.batch([data, label], batch_size=1000)
    
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        print(coord)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            i = 0
            while not coord.should_stop():
                i = i + 1
                XData, YData = sess.run([X_batch, Y_batch])
        finally:
            coord.request_stop()
            coord.join(threads)

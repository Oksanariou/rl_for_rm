from keras import backend as K
import tensorflow as tf
import timeit

with K.tf.device('/cpu:1'):
    config = tf.ConfigProto(device_count = {'CPU' : 1, 'GPU' : 1})

    random_image_gpu = tf.random_normal((100, 100, 100, 3))
    net_gpu = tf.layers.conv2d(random_image_gpu, 32, 7)
    net_gpu = tf.reduce_sum(net_gpu)

    session = tf.Session(config=config)

    K.set_session(session)

def gpu():
    session.run(net_gpu)

gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
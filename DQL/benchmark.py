from keras import backend as K
import tensorflow as tf
import timeit

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with K.tf.device('/gpu:0'):
    random_image_gpu = tf.random_normal((100, 100, 100, 3))
    net_gpu = tf.layers.conv2d(random_image_gpu, 32, 7)
    net_gpu = tf.reduce_sum(net_gpu)

with K.tf.device('/cpu:0'):
    random_image_cpu = tf.random_normal((100, 100, 100, 3))
    net_cpu = tf.layers.conv2d(random_image_cpu, 32, 7)
    net_cpu = tf.reduce_sum(net_cpu)


sess = tf.Session(config=config)

def gpu():
    sess.run(net_gpu)

def cpu():
    sess.run(net_cpu)

gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)

cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
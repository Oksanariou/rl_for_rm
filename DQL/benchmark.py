from keras.layers import K
import timeit

config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True

with K.tf.device('/gpu:0'):
    random_image_gpu = K.tf.random_normal((100, 100, 100, 3))
    net_gpu = K.tf.layers.conv2d(random_image_gpu, 32, 7)
    net_gpu = K.tf.reduce_sum(net_gpu)

sess = K.tf.Session(config=config)

def gpu():
    sess.run(net_gpu)

gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)

sess.close()
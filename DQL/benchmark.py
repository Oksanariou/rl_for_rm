from keras import backend as K
import tensorflow as tf
import timeit
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

A = np.random.rand(100, 5000)
B = np.random.rand(5000, 6000)

# with K.tf.device('/gpu:0'):
#     # random_image_gpu = K.random_normal((100, 100, 100, 3))
#     # net_gpu = K.conv2d(random_image_gpu, 32, 7)
#     # net_gpu = K.sum(net_gpu)
#     x = K.variable(value=A)
#     y = K.variable(value=B)
#
#     z = K.dot(x, y)
#
#     # Here you need to use K.eval() instead of z.eval() because this uses the backend session
#     K.eval(z)
#
# with K.tf.device('/cpu:0'):
#     # random_image_cpu = K.random_normal((100, 100, 100, 3))
    # net_cpu = K.conv2d(random_image_cpu, 32, 7)
    # net_cpu = K.sum(net_cpu)
def multi():
    x = K.variable(value=A)
    y = K.variable(value=B)

    z = K.dot(x, y)

        # Here you need to use K.eval() instead of z.eval() because this uses the backend session
    return K.eval(z)


sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
K.set_session(sess)
#
# def gpu():
#     sess.run(net_gpu)
#
# def cpu():
#     sess.run(net_cpu)

gpu_time = timeit.timeit('multi()', number=10, setup="from __main__ import multi")
print(gpu_time)

# cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
# print(cpu_time)
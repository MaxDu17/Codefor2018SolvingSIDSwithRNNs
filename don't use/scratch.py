import tensorflow as tf
import numpy as np
I = tf.placeholder(tf.float32, shape=[1, 3], name='I')  # input
W = tf.Variable(tf.zeros(shape=[3, 2]), dtype=tf.float32, name='W')  # weights
b = tf.Variable(tf.zeros(shape=[2]), dtype=tf.float32, name='b')  # biases
O = tf.nn.relu(tf.matmul(I, W) + b, name='O')  # activation / output

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    # save the graph
    tf.train.write_graph(sess.graph_def, '.', 'scratch/tfdroid.pbtxt')

    # normally you would do some training here
    # but fornow we will just assign something to W
    sess.run(tf.assign(W, [[1, 2], [4, 5], [7, 8]]))
    sess.run(tf.assign(b, [1, 1]))
    test = np.zeros(shape=[1,3])
    print(sess.run(O, feed_dict = {I:test}))
    # save a checkpoint file, which will store the above assignment
    saver.save(sess, 'scratch/tfdroid')
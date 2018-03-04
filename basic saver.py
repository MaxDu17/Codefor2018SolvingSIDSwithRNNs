import tensorflow as tf

a = tf.Variable(3)
b = tf.placeholder(tf.int32, name = "inplace")

final = tf.add(a, b)




import tensorflow as tf

a = tf.Variable(3)
b = tf.placeholder(tf.int32, name = "inplace")

final = tf.add(a, b)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(final, feed_dict = {b:12})
    saver.save(sess, "scratch/test",global_step = 1)
print(result)

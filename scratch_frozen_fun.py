import tensorflow as tf
import numpy as np
pbfilename = "scratch/frozen_modelv1.pb"



with tf.gfile.GFile(pbfilename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def,
                        input_map = None,
                        return_elements = None,
                        name = "")
output = graph.get_tensor_by_name("O:0")
I = graph.get_tensor_by_name("I:0")
with tf.Session(graph=graph) as sess:
    test = np.zeros(shape=[1,3])
    print(sess.run(output, feed_dict = {I:test}))

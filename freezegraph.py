import os, argparse

import tensorflow as tf

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

dir = os.path.dirname(os.path.realpath(__file__))

saver = tf.train.import_meta_graph('Best/rough_run-9500.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "Best/rough_run-9500")


#output_node_names = tf.all_variables()
output_node_names = ["W_In", "W_Out","W_Hidd","B_In", "B_Out", "B_Hidd"]
output_graph_def = tf.graph_util.convert_variables_to_constants(
    sess,  # The session
    input_graph_def,  # input_graph_def is useful for retrieving the nodes
    output_node_names

)


output_graph = "Best/modelv1.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())

sess.close()
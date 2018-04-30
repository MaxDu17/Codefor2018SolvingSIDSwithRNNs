import tensorflow as tf
import os
import numpy as np
from parse_data import DataParse as DP
parser = DP()
class WholeGraph:
    class Hyperparameters:
        input = 512
        first = 128
        compression = 50
        third = 128
        output = 512

        BATCH_NUMBER = 300
        LEARNING_RATE = 0.08
        EPOCHS = 2001
        TOTAL = 800

    def make_matrix_from_data(self, data):

        pbfilename = "GraphV6/Audoencoder_frozen.pb"
        with tf.gfile.GFile(pbfilename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def,
                                input_map = None,
                                return_elements = None,
                                name = "")
            input = graph.get_tensor_by_name("placeholders/input_and_label_placeholder:0")
            output = graph.get_tensor_by_name("encoding/compression:0")

        with tf.Session(graph=graph) as sess:
            whole_data = list()
            sliced = parser.prepare_data_autoencoder_from_raw(data)
            for slice in sliced:
                slice = np.reshape(slice, [1, 512])
                compressed_data = sess.run(output, feed_dict = {input:slice})
                whole_data.append(compressed_data)
            return whole_data

    def make_matrix_from_name(self, name):
        pbfilename = "GraphV6/Audoencoder_frozen.pb"
        with tf.gfile.GFile(pbfilename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def,
                                input_map = None,
                                return_elements = None,
                                name = "")
            input = graph.get_tensor_by_name("placeholders/input_and_label_placeholder:0")
            output = graph.get_tensor_by_name("encoding/compression:0")

        with tf.Session(graph=graph) as sess:
            whole_data = list()
            sliced = parser.prepare_data_autoencoder(name)

            for slice in sliced:
                slice = np.reshape(slice, [1, 512])
                compressed_data = sess.run(output, feed_dict = {input:slice})
                whole_data.append(compressed_data)
            return whole_data


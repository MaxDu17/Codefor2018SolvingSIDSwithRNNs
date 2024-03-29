import tensorflow as tf
import os
import numpy as np
from make_sets import Setmaker as SM
class WholeGraph:
    class Hyperparameters:
        INPUT_LAYER = 43
        HIDDEN_LAYER = 75 #Modify??
        OUTPUT_LAYER = 3
        NUM_EPOCHS = 10000
        #NUM_EPOCHS = 1
        BATCH_NUMBER = 240
        LEARNING_RATE = 0.1
        VALIDATION_NUMBER = 30
        TEST_NUMBER = 30

    def make_prediction(self, data):
        HYP = self.Hyperparameters()
        pbfilename = "GraphV5/GRAPHS/GraphV5_frozen.pb"
        with tf.gfile.GFile(pbfilename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def,
                                input_map = None,
                                return_elements = None,
                                name = "")
            input = graph.get_tensor_by_name("placeholders/input_placeholder:0")
            output = graph.get_tensor_by_name("prediction_and_loss/output:0")
            last_hidd = graph.get_tensor_by_name("placeholders/previous_hidden_layer_placeholder:0")
            next_hidd_layer = graph.get_tensor_by_name("hidden_propagation/hidden_layer_propagation:0")

        with tf.Session(graph=graph) as sess:
            output_prediction_ = []
            counter = 0
            first = True
            input_array=data
            for slice in input_array:
                slice = np.reshape(slice, [1, HYP.INPUT_LAYER])
                if counter == 15:
                     output_prediction_ = sess.run(output, feed_dict=
                    {
                        input: slice,
                        last_hidd: prev_hidd_layer_
                    })
                else:
                    if (first):
                        prev_hidd_layer_ = np.zeros(shape=HYP.HIDDEN_LAYER)
                        prev_hidd_layer_ = np.reshape(prev_hidd_layer_, [1, HYP.HIDDEN_LAYER])
                        first = False

                    next_hidd_layer_ = sess.run(next_hidd_layer, feed_dict=
                    {
                        input: slice,
                        last_hidd: prev_hidd_layer_
                    })
                    prev_hidd_layer_ = next_hidd_layer_
                    counter += 1

            return output_prediction_

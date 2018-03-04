import os
import sys
import tensorflow as tf
class Hyperparameters:
    INPUT_LAYER = 43
    HIDDEN_LAYER = 100 #Modify??
    OUTPUT_LAYER = 3
    NUM_EPOCHS = 10000
    #NUM_EPOCHS = 1
    BATCH_NUMBER = 240
    LEARNING_RATE = 0.1
    VALIDATION_NUMBER = 30
    TEST_NUMBER = 30
HYP = Hyperparameters()
def create_inference_graph():
    W_In = tf.Variable(tf.random_normal(shape=[HYP.INPUT_LAYER, HYP.HIDDEN_LAYER], stddev=0.1, mean=0),
                       name="W_In")
    W_Hidd = tf.Variable(tf.random_normal(shape=[HYP.HIDDEN_LAYER, HYP.HIDDEN_LAYER], stddev=0.1, mean=0),
                         name="W_Hidd")
    W_Out = tf.Variable(tf.random_normal(shape=[HYP.HIDDEN_LAYER, HYP.OUTPUT_LAYER], stddev=0.1, mean=0), name="W_Out")

    B_In = tf.Variable(tf.zeros(HYP.HIDDEN_LAYER), name="B_In")
    B_Hidd = tf.Variable(tf.zeros(HYP.HIDDEN_LAYER), name="B_Hidd")
    B_Out = tf.Variable(tf.zeros(HYP.OUTPUT_LAYER), name="B_Out")
    X = tf.placeholder(shape=[1, HYP.INPUT_LAYER], name="input_placeholder", dtype=tf.float32)
    last_hidd = tf.placeholder(shape=[1, HYP.HIDDEN_LAYER], name="previous_hidden_layer_placeholder",
                               dtype=tf.float32)
    hidd_layer = tf.matmul(X, W_In)
    hidd_layer = tf.add(hidd_layer, B_In)
    propagated_prev_hidd_layer = tf.matmul(last_hidd, W_Hidd)
    propagated_prev_hidd_layer = tf.add(propagated_prev_hidd_layer, B_Hidd)
    concat_hidd_layer = tf.add(hidd_layer, propagated_prev_hidd_layer, name = "next_hidd_layer")
    concat_hidd_layer = tf.sigmoid(concat_hidd_layer)
    next_hidd_layer = concat_hidd_layer
    output_logit = tf.matmul(concat_hidd_layer, W_Out)
    output_logit = tf.add(output_logit, B_Out)
    output_prediction = tf.nn.softmax(output_logit, name = "labels_softmax")

create_inference_graph()
with tf.Session() as sess:
    #saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('Best/rough_run-9500.meta', clear_devices=True)
    saver.restore(sess, "Best/rough_run-9500")
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ['labels_softmax']
    )
    tf.train.write_graph(
        frozen_graph_def,
        os.path.dirname("Best/modelv1.pb"),
        os.path.basename("Best/modelv1.pb"),
        as_text=False
    )


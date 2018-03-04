import tensorflow as tf
import numpy as np
import random
import os
from make_sets import Setmaker as SM
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

class Information:
    INPUT_DIMENSIONS = 43
    INPUT_TIME_DIV = 0.125
    INPUT_SECTORS = 8
    SAMPLE_RATE = 4096


summed_loss =0
reported_sum_loss = 0
set_maker = SM()
HYP = Hyperparameters()
prediction_dictionary = {0:"inhale", 1:"exhale", 2:"unknown"}

W_In = tf.Variable(tf.random_normal(shape = [HYP.INPUT_LAYER,HYP.HIDDEN_LAYER], stddev = 0.1, mean = 0 ),name = "W_In")#note: this used to have a mean of zero, so check that
W_Hidd =tf.Variable(tf.random_normal(shape = [HYP.HIDDEN_LAYER,HYP.HIDDEN_LAYER], stddev = 0.1, mean =0  ),name = "W_Hidd")
W_Out =tf.Variable(tf.random_normal(shape = [HYP.HIDDEN_LAYER,HYP.OUTPUT_LAYER], stddev = 0.1, mean = 0),name = "W_Out")

B_In = tf.Variable(tf.zeros(HYP.HIDDEN_LAYER),name = "B_In")
B_Hidd = tf.Variable(tf.zeros(HYP.HIDDEN_LAYER), name = "B_Hidd")
B_Out = tf.Variable(tf.zeros(HYP.OUTPUT_LAYER), name = "B_Out")

with tf.name_scope("placeholders"):
    X = tf.placeholder(shape=[1,HYP.INPUT_LAYER],name = "input_placeholder",dtype = tf.float32)
    Y = tf.placeholder(shape=[1,HYP.OUTPUT_LAYER],name = "one_hot_labels_placeholder",dtype = tf.int8)
    last_hidd = tf.placeholder(shape=[1,HYP.HIDDEN_LAYER],name = "previous_hidden_layer_placeholder", dtype = tf.float32)

#with tf.device("/device:GPU:0"):
with tf.name_scope("input_propagation"):
    hidd_layer = tf.matmul(X,W_In)
    hidd_layer = tf.add(hidd_layer,B_In)

with tf.name_scope("hidden_propagation"):
    propagated_prev_hidd_layer = tf.matmul(last_hidd,W_Hidd)
    propagated_prev_hidd_layer = tf.add(propagated_prev_hidd_layer,B_Hidd)
    concat_hidd_layer = tf.add(hidd_layer,propagated_prev_hidd_layer)
    concat_hidd_layer = tf.sigmoid(concat_hidd_layer)
    next_hidd_layer = concat_hidd_layer

with tf.name_scope("logit_output"):
    output_logit = tf.matmul(concat_hidd_layer,W_Out)
    output_logit = tf.add(output_logit, B_Out)

with tf.name_scope("prediction_and_loss"):
    output_prediction = tf.nn.softmax(output_logit,name = "output")
    loss = tf.nn.softmax_cross_entropy_with_logits(logits = output_logit,labels=Y,name = "sparse_softmax_loss_function")
    total_loss = tf.reduce_mean(loss)
with tf.name_scope("train"):
    optimizer = tf.train.AdagradOptimizer(learning_rate=HYP.LEARNING_RATE).minimize(total_loss)

with tf.name_scope("summaries_and_saver"):

    tf.summary.histogram("W_Hidd", W_Hidd)
    tf.summary.histogram("W_In", W_In)
    tf.summary.histogram("W_Out", W_Out)

    tf.summary.histogram("B_Hidd", B_Hidd)
    tf.summary.histogram("B_In", B_In)
    tf.summary.histogram("B_Out", B_Out)

    tf.summary.scalar("Loss_at_sample",total_loss)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

with tf.Session() as sess:

    tf.train.write_graph(sess.graph_def, '.', 'Best/graph.pbtxt')



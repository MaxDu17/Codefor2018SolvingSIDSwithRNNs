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


file_name = "dataTEST/exhale/1.wav"
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

with tf.name_scope("prediction_output"):
    output_prediction = tf.nn.softmax(output_logit)


saver = tf.train.Saver()
with tf.Session() as sess:
    output_prediction_ = []
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'GRAPHCHECKPOINTS/rough_run-9500' )
    writer = tf.summary.FileWriter("GRAPHS_run/",sess.graph)
    input_array= set_maker.load_blind(name = file_name)
    counter = 0
    first = True
    for slice in input_array:
        slice = np.reshape(slice,[1,43])
        if counter == 15:
            next_hidd_layer_,output_prediction_ = sess.run([next_hidd_layer, output_prediction], feed_dict=
            {
                X: slice,
                last_hidd: prev_hidd_layer_
            })
        else:
            if(first):
                prev_hidd_layer_ = np.zeros(shape = HYP.HIDDEN_LAYER)
                prev_hidd_layer_ = np.reshape(prev_hidd_layer_,[1,100])
                first = False

            next_hidd_layer_ = sess.run(next_hidd_layer,feed_dict =
            {
                X: slice,
                last_hidd: prev_hidd_layer_
            })
            prev_hidd_layer_ = next_hidd_layer_
            counter+= 1

    print("this is the predicted matrix: ", output_prediction_)
    print("inhale: ", output_prediction_[0][0])
    print("exhale: ", output_prediction_[0][1])
    print("unknown: ", output_prediction_[0][2])
    print("winning prediction: ", prediction_dictionary[np.argmax(output_prediction_)])
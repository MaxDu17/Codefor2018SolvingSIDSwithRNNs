import tensorflow as tf
import numpy as np
import random
import os
from make_sets import Setmaker as SM
class Hyperparameters:
    INPUT_LAYER = 43
    HIDDEN_LAYER = 100 #Modify??
    OUTPUT_LAYER = 3
    NUM_EPOCHS = 5000
    BATCH_NUMBER = 240
    LEARNING_RATE = 0.1

class Information:
    INPUT_DIMENSIONS = 43
    INPUT_TIME_DIV = 0.125
    INPUT_SECTORS = 8
    SAMPLE_RATE = 4096



set_maker = SM()
HYP = Hyperparameters()

W_In = tf.Variable(tf.random_normal(shape = [HYP.INPUT_LAYER,HYP.HIDDEN_LAYER], stddev = 0.1 ))#note: this used to have a mean of zero, so check that
W_Hidd =tf.Variable(tf.random_normal(shape = [HYP.HIDDEN_LAYER,HYP.HIDDEN_LAYER], stddev = 0.1 ))
W_Out =tf.Variable(tf.random_normal(shape = [HYP.HIDDEN_LAYER,HYP.OUTPUT_LAYER], stddev = 0.1 ))

B_In = tf.Variable(tf.zeros(HYP.HIDDEN_LAYER))
B_Hidd = tf.Variable(tf.zeros(HYP.HIDDEN_LAYER))
B_Out = tf.Variable(tf.zeros(HYP.OUTPUT_LAYER))


X = tf.placeholder(shape=[1,HYP.INPUT_LAYER],name = "input")
Y = tf.placeholder(shape=[1,HYP.OUTPUT_LAYER],name = "one-hot labels")
last_hidd = tf.placeholder(shape=[1,HYP.HIDDEN_LAYER],name = "previous hidden layer")

hidd_layer = tf.matmul(X,W_In)
hidd_layer = tf.add(hidd_layer,B_In)

propagated_prev_hidd_layer = tf.matmul(last_hidd,W_Hidd)
propagated_prev_hidd_layer = tf.add(propagated_prev_hidd_layer,B_Hidd)

concat_hidd_layer = tf.add(hidd_layer,propagated_prev_hidd_layer)
concat_hidd_layer = tf.sigmoid(concat_hidd_layer)
next_hidd_layer = concat_hidd_layer

output_logit = tf.matmul(concat_hidd_layer,W_Out)
output_logit = tf.add(output_logit, B_Out)

output_prediction = tf.nn.softmax(output_logit)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = output_logit,labels=Y,name = "sparse softmax loss function")
optimizer = tf.train.AdagradOptimizer(learning_rate=HYP.LEARNING_RATE).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    set_maker.get_test_set()

    for epoch in range(HYP.NUM_EPOCHS):
        set_maker.load_next_epoch()

        for batch_number in range(HYP.BATCH_NUMBER):
            input_array,label = set_maker.load_next_train_sample(batch_number = batch_number)
            one_hot_label = set_maker.one_hot_from_label(label=label)

            for slice in input_array:

